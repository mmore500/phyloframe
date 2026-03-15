import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_delete_unifurcating_roots_polars import (
    alifestd_delete_unifurcating_roots_polars as alifestd_delete_unifurcating_roots_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_delete_unifurcating_roots_polars = enforce_dtype_stability_polars(
    alifestd_delete_unifurcating_roots_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_delete_unifurcating_roots_polars_simple(
    apply: typing.Callable,
):
    """Unifurcating root 0 (single child 1) is deleted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_delete_unifurcating_roots_polars(df_pl).lazy().collect()

    assert sorted(result["id"].to_list()) == [1, 2, 3]
    ancestor_list = result.sort("id")["ancestor_id"].to_list()
    assert ancestor_list == [1, 1, 1]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_delete_unifurcating_roots_polars_no_unifurcating(
    apply: typing.Callable,
):
    """Root with two children is not unifurcating, no change."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        ),
    )

    result = alifestd_delete_unifurcating_roots_polars(df_pl).lazy().collect()

    assert sorted(result["id"].to_list()) == [0, 1, 2]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_delete_unifurcating_roots_polars_single_node(
    apply: typing.Callable,
):
    """Single node is both root and leaf, not unifurcating."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_delete_unifurcating_roots_polars(df_pl).lazy().collect()

    assert result["id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_delete_unifurcating_roots_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe is handled."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_delete_unifurcating_roots_polars(df_pl).lazy().collect()
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_delete_unifurcating_roots_polars_non_contiguous_ids(
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
        alifestd_delete_unifurcating_roots_polars(df_pl).lazy().collect()
