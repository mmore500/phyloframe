import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_is_working_format_polars import (
    alifestd_is_working_format_polars as alifestd_is_working_format_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_is_working_format_polars = enforce_dtype_stability_polars(
    alifestd_is_working_format_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_working_format(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
            }
        )
    )
    assert alifestd_is_working_format_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_no_ancestor_id(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[1]"],
            }
        )
    )
    assert not alifestd_is_working_format_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_contiguous(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 4],
                "ancestor_id": [0, 0, 2],
            }
        )
    )
    assert not alifestd_is_working_format_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_not_topologically_sorted(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [1, 0, 2],
                "ancestor_id": [0, 0, 1],
            }
        )
    )
    assert not alifestd_is_working_format_polars(df)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_root(apply: typing.Callable):
    df = apply(pl.DataFrame({"id": [0], "ancestor_id": [0]}))
    assert alifestd_is_working_format_polars(df)
