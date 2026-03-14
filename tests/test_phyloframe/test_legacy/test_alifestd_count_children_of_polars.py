import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_count_children_of_polars import (
    alifestd_count_children_of_polars as alifestd_count_children_of_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_count_children_of_polars = enforce_dtype_stability_polars(
    alifestd_count_children_of_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_root_with_two_children(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        )
    )
    assert alifestd_count_children_of_polars(df, 0) == 2


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_leaf_node(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
            }
        )
    )
    assert alifestd_count_children_of_polars(df, 1) == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_root_no_children(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        )
    )
    assert alifestd_count_children_of_polars(df, 0) == 0


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_parent_not_found(apply: typing.Callable):
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1],
                "ancestor_id": [0, 0],
            }
        )
    )
    with pytest.raises(ValueError, match="Parent 99 not found"):
        alifestd_count_children_of_polars(df, 99)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_with_ancestor_list(apply: typing.Callable):
    """Test that ancestor_id is inferred from ancestor_list."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[none]", "[0]", "[0]"],
            }
        )
    )
    assert alifestd_count_children_of_polars(df, 0) == 2
