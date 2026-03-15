import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_topological_sort_polars import (
    alifestd_topological_sort_polars as alifestd_topological_sort_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_topological_sort_polars = enforce_dtype_stability_polars(
    alifestd_topological_sort_polars_
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_topological_sort_polars_already_sorted(
    apply: typing.Callable,
):
    """Already sorted data stays sorted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_topological_sort_polars(df_pl).lazy().collect()

    assert len(result) == 5
    # All ancestors appear before their descendants
    id_list = result["id"].to_list()
    ancestor_list = result["ancestor_id"].to_list()
    for i, (nid, aid) in enumerate(zip(id_list, ancestor_list)):
        if nid != aid:  # not a root
            assert aid in id_list[:i]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_topological_sort_polars_reversed(
    apply: typing.Callable,
):
    """Reversed data gets properly sorted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 0, 1, 1],
            }
        ),
    )

    result = alifestd_topological_sort_polars(df_pl).lazy().collect()

    id_list = result["id"].to_list()
    ancestor_list = result["ancestor_id"].to_list()
    for i, (nid, aid) in enumerate(zip(id_list, ancestor_list)):
        if nid != aid:
            assert aid in id_list[:i]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_topological_sort_polars_single_node(
    apply: typing.Callable,
):
    """Single node is trivially sorted."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = alifestd_topological_sort_polars(df_pl).lazy().collect()
    assert result["id"].to_list() == [0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_topological_sort_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe is handled."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = alifestd_topological_sort_polars(df_pl).lazy().collect()
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_topological_sort_polars_non_contiguous_ids(
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
        alifestd_topological_sort_polars(df_pl).lazy().collect()
