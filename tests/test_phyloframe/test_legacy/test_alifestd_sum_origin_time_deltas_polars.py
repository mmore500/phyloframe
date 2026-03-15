import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_sum_origin_time_deltas_polars import (
    alifestd_sum_origin_time_deltas_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple_tree(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "origin_time": [0.0, 1.0, 2.0],
            }
        )
    )
    result = alifestd_sum_origin_time_deltas_polars(df)
    assert result == pytest.approx(3.0)  # 0 + 1 + 2 = 3


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [5.0],
            }
        )
    )
    result = alifestd_sum_origin_time_deltas_polars(df)
    assert result == pytest.approx(0.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "origin_time": [0.0, 1.0, 3.0],
            }
        )
    )
    alifestd_sum_origin_time_deltas_polars(df)
