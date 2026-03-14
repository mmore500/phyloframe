import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_ot_mrca_polars import (
    alifestd_mark_ot_mrca_polars as alifestd_mark_ot_mrca_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_ot_mrca_polars = enforce_dtype_stability_polars(
    alifestd_mark_ot_mrca_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple(apply: typing.Callable):
    """Simple star tree: root 0 with two leaves at origin_time=1."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "origin_time": [0, 1, 1],
            }
        ),
    )

    result = alifestd_mark_ot_mrca_polars(df).lazy().collect().sort("id")

    result_map = {
        row["id"]: row
        for row in result.iter_rows(named=True)
    }

    # At time 1 (nodes 1 and 2): active lineages are leaves 1, 2.
    # Their MRCA is node 0.
    assert result_map[1]["ot_mrca_id"] == 0
    assert result_map[1]["ot_mrca_time_of"] == 0
    assert result_map[1]["ot_mrca_time_since"] == 1

    assert result_map[2]["ot_mrca_id"] == 0
    assert result_map[2]["ot_mrca_time_of"] == 0
    assert result_map[2]["ot_mrca_time_since"] == 1

    # At time 0 (node 0): MRCA is node 0 itself.
    assert result_map[0]["ot_mrca_id"] == 0
    assert result_map[0]["ot_mrca_time_of"] == 0
    assert result_map[0]["ot_mrca_time_since"] == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    """Empty dataframe should return with three added columns."""
    df = apply(
        pl.DataFrame(
            {
                "id": [],
                "ancestor_id": [],
                "origin_time": [],
            },
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "origin_time": pl.Float64,
            },
        ),
    )

    result = alifestd_mark_ot_mrca_polars(df).lazy().collect()
    assert result.is_empty()
    assert "ot_mrca_id" in result.columns
    assert "ot_mrca_time_of" in result.columns
    assert "ot_mrca_time_since" in result.columns


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    """Non-contiguous ids should raise NotImplementedError."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "origin_time": [0, 1, 2],
            }
        ),
    )

    with pytest.raises(NotImplementedError):
        alifestd_mark_ot_mrca_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_linear_chain(apply: typing.Callable):
    """Linear chain: 0 -> 1 -> 2 (only node 2 is a leaf)."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [0, 1, 2],
            }
        ),
    )

    result = alifestd_mark_ot_mrca_polars(df).lazy().collect().sort("id")

    result_map = {
        row["id"]: row
        for row in result.iter_rows(named=True)
    }

    # At time 2 (node 2, only leaf): MRCA = 2 itself
    assert result_map[2]["ot_mrca_id"] == 2
    assert result_map[2]["ot_mrca_time_of"] == 2
    assert result_map[2]["ot_mrca_time_since"] == 0

    # At time 1 (node 1): running mrca from time 2 was 2,
    # combined with earliest id at time 1 (which is 1).
    # Lineages: {1, 2} -> ancestor of 2 is 1, so MRCA = 1
    assert result_map[1]["ot_mrca_id"] == 1
    assert result_map[1]["ot_mrca_time_of"] == 1
    assert result_map[1]["ot_mrca_time_since"] == 0

    # At time 0 (node 0): running mrca from time 1 was 1,
    # combined with earliest id at time 0 (which is 0).
    # Lineages: {0, 1} -> ancestor of 1 is 0, so MRCA = 0
    assert result_map[0]["ot_mrca_id"] == 0
    assert result_map[0]["ot_mrca_time_of"] == 0
    assert result_map[0]["ot_mrca_time_since"] == 0
