import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_collapse_trunk_polars import (
    alifestd_collapse_trunk_polars as alifestd_collapse_trunk_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_collapse_trunk_polars = enforce_dtype_stability_polars(
    alifestd_collapse_trunk_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_collapse_trunk_polars_simple(
    apply: typing.Callable,
):
    """Trunk nodes are collapsed to oldest root."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 1, 1, 2],
                "is_trunk": [True, True, False, False, False],
            }
        ),
    )

    result = alifestd_collapse_trunk_polars(df_pl).lazy().collect()

    # Trunk nodes collapsed; non-trunk nodes whose ancestor was trunk
    # are reparented to oldest root 0. Node 4's ancestor (2) is non-trunk.
    result_sorted = result.sort("id")
    expected = {0: 0, 2: 0, 3: 0, 4: 2}
    for row in result_sorted.iter_rows(named=True):
        assert row["ancestor_id"] == expected[row["id"]]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_collapse_trunk_polars_no_trunk(
    apply: typing.Callable,
):
    """All is_trunk=False with single trunk node, no change."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "is_trunk": [False, False, False],
            }
        ),
    )

    result = alifestd_collapse_trunk_polars(df_pl).lazy().collect()

    assert len(result) == 3


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_collapse_trunk_polars_missing_is_trunk(
    apply: typing.Callable,
):
    """Missing is_trunk column raises ValueError."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        ),
    )

    with pytest.raises(ValueError):
        alifestd_collapse_trunk_polars(df_pl).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_collapse_trunk_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe with is_trunk column is handled."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": [], "is_trunk": []},
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "is_trunk": pl.Boolean,
            },
        ),
    )

    result = alifestd_collapse_trunk_polars(df_pl).lazy().collect()
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_collapse_trunk_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "is_trunk": [True, False, False],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_collapse_trunk_polars(df_pl).lazy().collect()


def test_alifestd_collapse_trunk_polars_with_origin_time():
    """When origin_time is present, oldest root is selected by time."""
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 1, 1, 2],
            "is_trunk": [True, True, False, False, False],
            "origin_time": [0.0, 1.0, 2.0, 2.0, 3.0],
        }
    )

    result = alifestd_collapse_trunk_polars(df_pl).lazy().collect()
    result_sorted = result.sort("id")

    # Node 0 (origin_time=0.0) is oldest trunk root, kept
    # Node 1 (trunk) collapsed
    assert 0 in result_sorted["id"].to_list()
    assert 1 not in result_sorted["id"].to_list()


def test_alifestd_collapse_trunk_polars_single_trunk():
    """Single trunk node should return unchanged."""
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 0],
            "is_trunk": [True, False, False],
        }
    )

    result = alifestd_collapse_trunk_polars(df_pl).lazy().collect()
    assert len(result) == 3
