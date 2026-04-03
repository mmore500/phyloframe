import os
import typing

import numpy as np
import pandas as pd
import polars as pl
import pytest
from tqdm import tqdm

from phyloframe.legacy import (
    alifestd_calc_distance_matrix_asexual,
)
from phyloframe.legacy import (
    alifestd_calc_distance_matrix_polars as alifestd_calc_distance_matrix_polars_,
)
from phyloframe.legacy import (
    alifestd_to_working_format,
)

from ._impl import enforce_dtype_stability_polars

alifestd_calc_distance_matrix_polars = enforce_dtype_stability_polars(
    alifestd_calc_distance_matrix_polars_
)

assets_path = os.path.join(os.path.dirname(__file__), "assets")


@pytest.mark.parametrize(
    "phylogeny_df",
    [
        pd.read_csv(f"{assets_path}/nk_ecoeaselection.csv"),
        pd.read_csv(f"{assets_path}/nk_lexicaseselection.csv"),
        pd.read_csv(f"{assets_path}/nk_tournamentselection.csv"),
    ],
)
@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_big1_matches_pandas(
    phylogeny_df: pd.DataFrame, apply: typing.Callable
):
    """Verify polars result matches pandas distance matrix on real datasets."""
    phylogeny_df = phylogeny_df.copy()
    phylogeny_df = alifestd_to_working_format(phylogeny_df)
    df_pl = pl.from_pandas(phylogeny_df)

    expected = alifestd_calc_distance_matrix_asexual(phylogeny_df)
    actual = alifestd_calc_distance_matrix_polars(
        apply(df_pl), progress_wrap=tqdm
    )

    np.testing.assert_array_equal(
        np.isnan(expected), np.isnan(actual), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(actual), np.nan_to_num(expected))


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_simple1(apply: typing.Callable):
    """Test a simple 4-node tree matches known expected distance matrix."""
    # Tree:  0 -> 1 -> 2, 0 -> 3
    # origin_times: 0=0, 1=10, 2=20, 3=15
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 0],
                "origin_time": [0, 10, 20, 15],
            }
        )
    )
    expected = np.array(
        [
            [0.0, 10.0, 20.0, 15.0],
            [10.0, 0.0, 10.0, 25.0],
            [20.0, 10.0, 0.0, 35.0],
            [15.0, 25.0, 35.0, 0.0],
        ],
        dtype=np.float64,
    )
    actual = alifestd_calc_distance_matrix_polars(df_pl)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_custom_criterion(apply: typing.Callable):
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 0],
                "origin_time": [0, 10, 20, 15],
                "depth": [0.0, 1.0, 2.0, 1.0],
            }
        )
    )
    res = alifestd_calc_distance_matrix_polars(df_pl, criterion="depth")
    # MRCA of 2 and 3 is 0; depth distance = (2.0 + 1.0 - 2*0.0) = 3.0
    assert res[2, 3] == pytest.approx(3.0)
    assert res[3, 2] == pytest.approx(3.0)
    for i in range(4):
        assert res[i, i] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
@pytest.mark.parametrize(
    "phylogeny_df_pd",
    [
        pd.DataFrame(
            {
                "id": pd.array([], dtype="int64"),
                "ancestor_list": pd.array([], dtype="object"),
                "origin_time": pd.array([], dtype="float64"),
            }
        ),
        pd.DataFrame(
            {
                "id": pd.array([], dtype="int64"),
                "ancestor_id": pd.array([], dtype="int64"),
                "origin_time": pd.array([], dtype="float64"),
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_list": ["[None]"],
                "origin_time": [5.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "ancestor_list": ["[None]"],
                "origin_time": [5.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1],
                "ancestor_list": ["[None]", "[0]"],
                "origin_time": [0.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 0],
                "ancestor_list": ["[None]", "[0]", "[0]"],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_list": ["[None]", "[None]", "[1]"],
                "origin_time": [0.0, 5.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 1],
                "origin_time": [0.0, 5.0, 3.0, 10.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 0, 0],
                "origin_time": [0.0, 5.0, 8.0, 12.0],
            }
        ),
        pd.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[0]", "[0]"],
                "origin_time": [0.0, 5.0, 8.0, 12.0],
            }
        ),
    ],
)
def test_edge_cases(phylogeny_df_pd: pd.DataFrame, apply: typing.Callable):
    """Test edge cases match the pandas distance matrix implementation."""
    phylogeny_df_pd = phylogeny_df_pd.copy()

    expected = alifestd_calc_distance_matrix_asexual(phylogeny_df_pd)

    phylogeny_df_pd_wf = alifestd_to_working_format(phylogeny_df_pd.copy())
    df_pl = pl.from_pandas(phylogeny_df_pd_wf)

    actual = alifestd_calc_distance_matrix_polars(apply(df_pl))
    np.testing.assert_array_equal(
        np.isnan(actual), np.isnan(expected), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(actual), np.nan_to_num(expected))

    # idempotency check
    actual2 = alifestd_calc_distance_matrix_polars(apply(df_pl))
    np.testing.assert_array_equal(
        np.isnan(actual2), np.isnan(expected), err_msg="NaN pattern mismatch"
    )
    np.testing.assert_allclose(np.nan_to_num(actual2), np.nan_to_num(expected))


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_does_not_mutate_input(apply: typing.Callable):
    """Verify the input dataframe is not mutated."""
    df_pl = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 1, 0],
            "origin_time": [0, 10, 20, 15],
        }
    )
    original_columns = df_pl.columns[:]
    original_data = df_pl.clone()

    alifestd_calc_distance_matrix_polars(apply(df_pl))

    assert df_pl.columns == original_columns
    assert df_pl.equals(original_data)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_single_node(apply: typing.Callable):
    """Test single-node phylogeny: distance with self is 0."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [42.0],
            }
        )
    )
    result = alifestd_calc_distance_matrix_polars(df_pl)
    np.testing.assert_allclose(result, np.array([[0.0]], dtype=np.float64))


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_multiple_roots_nan(apply: typing.Callable):
    """Test phylogeny with multiple independent roots has NaN cross-tree."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [0.0, 5.0, 10.0],
            }
        )
    )
    result = alifestd_calc_distance_matrix_polars(df_pl)
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 1] == pytest.approx(0.0)
    assert result[2, 2] == pytest.approx(0.0)
    assert np.isnan(result[0, 1])
    assert np.isnan(result[1, 0])
    assert np.isnan(result[0, 2])
    assert np.isnan(result[1, 2])


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_symmetric(apply: typing.Callable):
    """Distance matrix should be symmetric."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 1, 0],
                "origin_time": [0, 10, 20, 15],
            }
        )
    )
    result = alifestd_calc_distance_matrix_polars(df_pl)
    np.testing.assert_allclose(result, result.T)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_uint64_ancestor_id(apply: typing.Callable):
    """Regression: UInt64 ancestor_id should be handled correctly."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": pl.Series([0, 1, 2, 3], dtype=pl.UInt64),
                "ancestor_id": pl.Series([0, 0, 1, 0], dtype=pl.UInt64),
                "origin_time": pl.Series(
                    [0.0, 10.0, 20.0, 15.0], dtype=pl.Float64
                ),
            }
        )
    )
    result = alifestd_calc_distance_matrix_polars(df_pl)
    assert result.dtype == np.float64
    assert result.shape == (4, 4)
    np.testing.assert_allclose(result, result.T)


@pytest.mark.parametrize(
    "apply",
    [
        pytest.param(lambda x: x, id="DataFrame"),
        pytest.param(lambda x: x.lazy(), id="LazyFrame"),
    ],
)
def test_with_ancestor_list_col(apply: typing.Callable):
    """Test that ancestor_list is correctly converted to ancestor_id."""
    df_pl = apply(
        pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_list": ["[None]", "[0]", "[0]", "[1]"],
                "origin_time": [0.0, 10.0, 5.0, 20.0],
            }
        )
    )
    result = alifestd_calc_distance_matrix_polars(df_pl)
    # Tree: 0->1->3, 0->2.
    # distance[0,0]=0, distance[1,2]: mrca=0; (10+5-2*0)=15
    assert result[0, 0] == pytest.approx(0.0)
    assert result[1, 2] == pytest.approx(15.0)
    assert result[2, 1] == pytest.approx(15.0)
    # Self-distances
    for i in range(4):
        assert result[i, i] == pytest.approx(0.0)
