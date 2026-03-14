import math
import typing

import polars as pl
import pytest

from phyloframe.legacy._alifestd_mark_clade_subtended_duration_ratio_sister_polars import (
    alifestd_mark_clade_subtended_duration_ratio_sister_polars as alifestd_mark_clade_subtended_duration_ratio_sister_polars_,
)

from ._impl import enforce_dtype_stability_polars

alifestd_mark_clade_subtended_duration_ratio_sister_polars = (
    enforce_dtype_stability_polars(
        alifestd_mark_clade_subtended_duration_ratio_sister_polars_,
    )
)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_ratio_sister_polars_simple_tree(
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

    max_desc_ot:     [6, 4, 6, 3, 4, 5, 6]
    ancestor_ot:     [0, 0, 0, 1, 1, 2, 2]
    eff_ancestor_ot: [0, 0, 0, 1, 1, 2, 2]  (root uses 0)
    clade_sub_dur:   [6, 4, 6, 2, 3, 3, 4]
    sister_id:       [0, 2, 1, 4, 3, 6, 5]

    ratios:
      0: 6/6 = 1.0 (root)
      1: 4/6 = 0.666...
      2: 6/4 = 1.5
      3: 2/3 = 0.666...
      4: 3/2 = 1.5
      5: 3/4 = 0.75
      6: 4/3 = 1.333...
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
        alifestd_mark_clade_subtended_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    vals = result["clade_subtended_duration_ratio_sister"].to_list()
    assert vals[0] == pytest.approx(1.0)
    assert vals[1] == pytest.approx(4.0 / 6.0)
    assert vals[2] == pytest.approx(6.0 / 4.0)
    assert vals[3] == pytest.approx(2.0 / 3.0)
    assert vals[4] == pytest.approx(3.0 / 2.0)
    assert vals[5] == pytest.approx(3.0 / 4.0)
    assert vals[6] == pytest.approx(4.0 / 3.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_ratio_sister_polars_single_node(
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
        alifestd_mark_clade_subtended_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    # root: sub_dur = 5 - 0 = 5; sister is self; 5/5 = 1.0
    assert result["clade_subtended_duration_ratio_sister"].to_list()[
        0
    ] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_ratio_sister_polars_empty(
    apply: typing.Callable,
):
    """Empty dataframe gets the column."""
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
        alifestd_mark_clade_subtended_duration_ratio_sister_polars(df_pl)
        .lazy()
        .collect()
    )

    assert "clade_subtended_duration_ratio_sister" in result.columns
    assert result.is_empty()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_ratio_sister_polars_non_contiguous_ids(
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
        alifestd_mark_clade_subtended_duration_ratio_sister_polars(
            df_pl,
        ).lazy().collect()


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_alifestd_mark_clade_subtended_duration_ratio_sister_polars_unsorted(
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
        alifestd_mark_clade_subtended_duration_ratio_sister_polars(
            df_pl,
        ).lazy().collect()
