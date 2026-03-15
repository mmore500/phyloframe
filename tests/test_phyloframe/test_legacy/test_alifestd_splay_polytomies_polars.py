import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_splay_polytomies_polars import (
    alifestd_splay_polytomies_polars as alifestd_splay_polytomies_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_splay_polytomies_polars = enforce_dtype_stability_polars(
    alifestd_splay_polytomies_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_splay_polytomies_polars_polytomy(
    apply: typing.Callable,
):
    """Root with 3 children (polytomy) is splayed into bifurcations."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
            }
        ),
    )

    result = alifestd_splay_polytomies_polars(df_pl).lazy().collect()

    assert len(result) == 5
    # Verify no node has more than 2 children
    ancestor_counts = (
        result.filter(pl.col("id") != pl.col("ancestor_id"))
        .group_by("ancestor_id")
        .agg(pl.col("id").count().alias("child_count"))
    )
    assert ancestor_counts["child_count"].max() <= 2


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_splay_polytomies_polars_already_bifurcating(
    apply: typing.Callable,
):
    """Already bifurcating tree stays the same."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        ),
    )

    result = alifestd_splay_polytomies_polars(df_pl).lazy().collect()

    assert len(result) == 3


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_splay_polytomies_polars_single_node(
    apply: typing.Callable,
):
    """Single node stays the same."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_splay_polytomies_polars(df_pl).lazy().collect()

    assert result["id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_splay_polytomies_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe is handled."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_splay_polytomies_polars(df_pl).lazy().collect()
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_splay_polytomies_polars_non_contiguous_ids(
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
        alifestd_splay_polytomies_polars(df_pl).lazy().collect()
