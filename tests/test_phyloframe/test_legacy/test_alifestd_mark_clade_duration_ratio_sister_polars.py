import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_clade_duration_ratio_sister_polars import (
    alifestd_mark_clade_duration_ratio_sister_polars as alifestd_mark_clade_duration_ratio_sister_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_clade_duration_ratio_sister_polars = (
    enforce_dtype_stability_polars(
        alifestd_mark_clade_duration_ratio_sister_polars_,
    )
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_duration_ratio_sister_polars_simple_tree(
    apply: typing.Callable,
):
    """Test a strictly bifurcating tree.

    Tree structure:
        0 (root, ot=0)
        +-- 1 (ot=1)
        |   +-- 3 (ot=3)
        |   +-- 4 (ot=4)
        +-- 2 (ot=2)
            +-- 5 (ot=5)
            +-- 6 (ot=6)

    max_desc_ot: [6, 4, 6, 3, 4, 5, 6]
    clade_dur:   [6, 3, 4, 0, 0, 0, 0]
    sister_id:   [0, 2, 1, 4, 3, 6, 5]

    ratios:
      0: 6/6 = 1.0 (root, sister is self)
      1: 3/4 = 0.75 (sister is 2, dur 4)
      2: 4/3 = 1.333... (sister is 1, dur 3)
      3: 0/0 = NaN (sister is 4, dur 0)
      4: 0/0 = NaN (sister is 3, dur 0)
      5: 0/0 = NaN (sister is 6, dur 0)
      6: 0/0 = NaN (sister is 5, dur 0)
    """
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3, 4, 5, 6],
                "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
                "origin_time": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        ),
    )

    result = (
        alifestd_mark_clade_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    vals = result["clade_duration_ratio_sister"].to_list()
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(0.75)
    assert vals[2] == pytest.approx(4.0 / 3.0)
    # leaf siblings have 0/0 = NaN
    import math

    assert math.isnan(vals[3])
    assert math.isnan(vals[4])
    assert math.isnan(vals[5])
    assert math.isnan(vals[6])


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_duration_ratio_sister_polars_single_node(
    apply: typing.Callable,
):
    """A single root node."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [5.0],
            }
        ),
    )

    result = (
        alifestd_mark_clade_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    import math

    # root: clade_dur=0, sister is self, 0/0 = NaN
    assert math.isnan(result["clade_duration_ratio_sister"].to_list()[0])


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_duration_ratio_sister_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets clade_duration_ratio_sister column."""
    df_pl = apply(
        pl.DataFrame(
            {"id": [], "ancestor_id": [], "origin_time": []},
            schema={
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "origin_time": pl.Float64,
            },
        ),
    )

    result = (
        alifestd_mark_clade_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    assert "clade_duration_ratio_sister" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_duration_ratio_sister_polars_non_contiguous_ids(
    apply: typing.Callable,
):
    """Verify NotImplementedError for non-contiguous ids."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_duration_ratio_sister_polars(
            df_pl
        ).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_duration_ratio_sister_polars_unsorted(
    apply: typing.Callable,
):
    """Verify NotImplementedError for topologically unsorted data."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 2, 0],
                "origin_time": [0.0, 1.0, 2.0],
            }
        ),
    )
    with pytest.raises(NotImplementedError):
        alifestd_mark_clade_duration_ratio_sister_polars(
            df_pl
        ).lazy().collect()
