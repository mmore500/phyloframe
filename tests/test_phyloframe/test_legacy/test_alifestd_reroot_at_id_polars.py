import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_reroot_at_id_polars import (
    alifestd_reroot_at_id_polars,
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_reroot_simple(apply: typing.Callable):
    """Reroot a simple tree at a leaf node."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 1, 1, 2],
            }
        ),
    )

    result = alifestd_reroot_at_id_polars(df, new_root_id=4)
    result = result.lazy().collect().sort("id")

    ancestor_map = {
        row["id"]: row["ancestor_id"] for row in result.iter_rows(named=True)
    }

    # node 4 should now be root (self-ancestor)
    assert ancestor_map[4] == 4
    # path from old root to new root is reversed:
    # was: 0 -> 1 -> 2 -> 4
    # now: 4 -> 2 -> 1 -> 0
    assert ancestor_map[2] == 4
    assert ancestor_map[1] == 2
    assert ancestor_map[0] == 1
    # node 3 was child of 1, should remain child of 1
    assert ancestor_map[3] == 1


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_reroot_at_root(apply: typing.Callable):
    """Rerooting at current root should not change ancestor relationships."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4],
                "ancestor_id": [0, 0, 1, 1, 2],
            }
        ),
    )

    result = alifestd_reroot_at_id_polars(df, new_root_id=0)
    result = result.lazy().collect().sort("id")

    ancestor_map = {
        row["id"]: row["ancestor_id"] for row in result.iter_rows(named=True)
    }

    assert ancestor_map[0] == 0
    assert ancestor_map[1] == 0
    assert ancestor_map[2] == 1
    assert ancestor_map[3] == 1
    assert ancestor_map[4] == 2


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_non_contiguous_ids(apply: typing.Callable):
    """Non-contiguous ids should be handled automatically."""
    df = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
            }
        ),
    )

    alifestd_reroot_at_id_polars(df, new_root_id=5)
