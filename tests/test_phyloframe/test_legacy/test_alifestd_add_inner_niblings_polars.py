import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_add_inner_niblings_polars import (
    alifestd_add_inner_niblings_polars as alifestd_add_inner_niblings_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_add_inner_niblings_polars = enforce_dtype_stability_polars(
    alifestd_add_inner_niblings_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
@pytest.mark.xfail(
    reason="knuckle insertion breaks topological sort order, "
    "causing alifestd_mark_node_depth_polars to raise "
    "NotImplementedError",
    raises=NotImplementedError,
    strict=True,
)
def test_simple(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_add_inner_niblings_polars(df).lazy().collect()

    # Result should have more rows than input (knuckles + niblings added)
    assert len(result) > 5

    # All new nibling rows should have is_leaf=True
    original_ids = {0, 1, 2, 3, 4}
    nibling_rows = result.filter(
        ~pl.col("id").is_in(list(original_ids)) & pl.col("is_leaf")
    )
    assert len(nibling_rows) > 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_empty(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_add_inner_niblings_polars(df).lazy().collect()
    assert result.is_empty()


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
            }
        ),
    )

    with pytest.raises(NotImplementedError):
        alifestd_add_inner_niblings_polars(df).lazy().collect()
