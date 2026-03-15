import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_clade_nodecount_ratio_sister_polars import (
    alifestd_mark_clade_nodecount_ratio_sister_polars as alifestd_mark_clade_nodecount_ratio_sister_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_clade_nodecount_ratio_sister_polars = (
    enforce_dtype_stability_polars(
        alifestd_mark_clade_nodecount_ratio_sister_polars_,
    )
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a strictly bifurcating tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   +-- 4
        +-- 2
            +-- 5
            +-- 6

    num_descendants: [6, 2, 2, 0, 0, 0, 0]
    sister_id:       [0, 2, 1, 4, 3, 6, 5]

    ratios = (num_desc + 1) / (num_desc[sister] + 1):
      0: 7/7 = 1.0
      1: 3/3 = 1.0
      2: 3/3 = 1.0
      3: 1/1 = 1.0
      4: 1/1 = 1.0
      5: 1/1 = 1.0
      6: 1/1 = 1.0
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
            }
        ),
    )

    result = (
        alifestd_mark_clade_nodecount_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    assert result["clade_nodecount_ratio_sister"].to_list() == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_asymmetric(
    apply: typing.Callable,
):
    """Test asymmetric bifurcating tree.

    Tree structure:
        0 (root)
        +-- 1
        |   +-- 3
        |   |   +-- 5
        |   |   +-- 6
        |   +-- 4
        +-- 2

    num_descendants: [6, 4, 0, 2, 0, 0, 0]
    sister_id:       [0, 2, 1, 4, 3, 6, 5]

    ratios = (nd+1)/(nd[sister]+1):
      0: 7/7 = 1.0
      1: 5/1 = 5.0
      2: 1/5 = 0.2
      3: 3/1 = 3.0
      4: 1/3 = 0.333...
      5: 1/1 = 1.0
      6: 1/1 = 1.0
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "ancestor_id": [0, 0, 0, 1, 1, 3, 3],
            }
        ),
    )

    result = (
        alifestd_mark_clade_nodecount_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    vals = result["clade_nodecount_ratio_sister"].to_list()
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(5.0)
    assert vals[2] == pytest.approx(0.2)
    assert vals[3] == pytest.approx(3.0)
    assert vals[4] == pytest.approx(1.0 / 3.0)
    assert vals[5] == pytest.approx(1.0)
    assert vals[6] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_single_node(
    apply: typing.Callable,
):
    """A single root node."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            }
        ),
    )

    result = (
        alifestd_mark_clade_nodecount_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    assert result["clade_nodecount_ratio_sister"].to_list() == [1.0]


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets the column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": []},
            schema={"id": pl.Int64, "ancestor_id": pl.Int64},
        ),
    )

    result = (
        alifestd_mark_clade_nodecount_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    assert "clade_nodecount_ratio_sister" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_non_contiguous_ids(
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
        alifestd_mark_clade_nodecount_ratio_sister_polars(
            df_pl,
        ).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_nodecount_ratio_sister_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_nodecount_ratio_sister_polars(
            df_pl,
        ).lazy().collect()
