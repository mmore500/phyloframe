import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from phyloframe.legacy._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from phyloframe.legacy._alifestd_to_working_format_polars import (
    alifestd_to_working_format_polars as alifestd_to_working_format_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_to_working_format_polars = enforce_dtype_stability_polars(
    alifestd_to_working_format_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_already_working(
    apply: typing.Callable,
):
    """Already-working data is left in working format."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert len(result) == 5
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_unsorted(
    apply: typing.Callable,
):
    """Topologically unsorted, contiguous-id data gets sorted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [4, 3, 2, 1, 0],
                "ancestor_id": [1, 1, 0, 0, 0],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert len(result) == 5
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Non-contiguous ids get reassigned to row indices."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [10, 20, 30, 40, 50],
                "ancestor_id": [10, 10, 10, 20, 20],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert len(result) == 5
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_unsorted_non_contiguous(
    apply: typing.Callable,
):
    """Unsorted, non-contiguous data is fully normalized."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [50, 40, 30, 20, 10],
                "ancestor_id": [20, 20, 10, 10, 10],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert len(result) == 5
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_single_node(
    apply: typing.Callable,
):
    """Single node is trivially in working format."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert result["id"].to_list() == [0]
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_adds_ancestor_id(
    apply: typing.Callable,
):
    """Asexual data with only ancestor_list gets an ancestor_id column."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_list": ["[none]", "[0]", "[0]", "[1]", "[1]"],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert "ancestor_id" in result.columns
    assert "ancestor_list" not in result.columns
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_drops_ancestor_list(
    apply: typing.Callable,
):
    """By default, ancestor_list is dropped from the result."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [10, 20, 30, 40, 50],
                "ancestor_id": [10, 10, 10, 20, 20],
                "ancestor_list": ["[none]", "[10]", "[10]", "[20]", "[20]"],
            }
        ),
    )

    result = alifestd_to_working_format_polars(df_pl).lazy().collect()

    assert "ancestor_list" not in result.columns
    assert "ancestor_id" in result.columns
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_keep_ancestor_list_absent(
    apply: typing.Callable,
):
    """keep_ancestor_list=True is a no-op when input had no ancestor_list."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [10, 20, 30, 40, 50],
                "ancestor_id": [10, 10, 10, 20, 20],
            }
        ),
    )

    result = (
        alifestd_to_working_format_polars(df_pl, keep_ancestor_list=True)
        .lazy()
        .collect()
    )

    assert "ancestor_list" not in result.columns
    assert "ancestor_id" in result.columns
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_to_working_format_polars_keep_ancestor_list(
    apply: typing.Callable,
):
    """When keep_ancestor_list=True, ancestor_list is regenerated."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [10, 20, 30, 40, 50],
                "ancestor_id": [10, 10, 10, 20, 20],
                "ancestor_list": ["[none]", "[10]", "[10]", "[20]", "[20]"],
            }
        ),
    )

    result = (
        alifestd_to_working_format_polars(df_pl, keep_ancestor_list=True)
        .lazy()
        .collect()
    )

    assert "ancestor_list" in result.columns
    assert "ancestor_id" in result.columns
    assert alifestd_has_contiguous_ids_polars(result)
    assert alifestd_is_topologically_sorted_polars(result)
    # root has [none], rest reference reassigned ancestor ids
    ids = result["id"].to_list()
    ancestor_ids = result["ancestor_id"].to_list()
    ancestor_lists = result["ancestor_list"].to_list()
    for nid, aid, alist in zip(ids, ancestor_ids, ancestor_lists):
        if nid == aid:
            assert alist == "[none]"
        else:
            assert alist == f"[{aid}]"
