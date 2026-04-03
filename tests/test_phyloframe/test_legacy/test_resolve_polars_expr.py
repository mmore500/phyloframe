import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._alifestd_calc_distance_matrix_polars import (
    alifestd_calc_distance_matrix_polars,
)
from phyloframe.legacy._alifestd_downsample_tips_canopy_polars import (
    alifestd_downsample_tips_canopy_polars,
)
from phyloframe.legacy._alifestd_find_pair_distance_polars import (
    alifestd_find_pair_distance_polars,
)
from phyloframe.legacy._alifestd_mark_sample_tips_canopy_polars import (
    alifestd_mark_sample_tips_canopy_polars,
)
from phyloframe.legacy._alifestd_sort_children_polars import (
    alifestd_sort_children_polars,
)


def _make_simple_phylogeny() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "ancestor_id": [0, 0, 0, 1, 1],
            "origin_time": [0, 1, 2, 3, 4],
        }
    )


def test_sort_children_polars_expr_matches_string():
    df = _make_simple_phylogeny()
    result_str = alifestd_sort_children_polars(df, "origin_time")
    result_expr = alifestd_sort_children_polars(
        df,
        pl.col("origin_time"),
    )
    assert result_str.equals(result_expr)


def test_sort_children_polars_expr_computed():
    df = _make_simple_phylogeny()
    result = alifestd_sort_children_polars(
        df,
        pl.col("origin_time") * -1,
        reverse=True,
    )
    assert result.shape == df.shape


def test_mark_sample_tips_canopy_polars_expr_matches_string():
    df = _make_simple_phylogeny()
    result_str = alifestd_mark_sample_tips_canopy_polars(
        df,
        criterion="origin_time",
    )
    result_expr = alifestd_mark_sample_tips_canopy_polars(
        df,
        criterion=pl.col("origin_time"),
    )
    assert result_str.equals(result_expr)


def test_mark_sample_tips_canopy_polars_expr_computed():
    df = _make_simple_phylogeny()
    result = alifestd_mark_sample_tips_canopy_polars(
        df,
        n_sample=2,
        criterion=pl.col("origin_time") * 10,
    )
    assert "alifestd_mark_sample_tips_canopy_polars" in result.columns


def test_find_pair_distance_polars_expr_matches_string():
    df = _make_simple_phylogeny()
    dist_str = alifestd_find_pair_distance_polars(
        df,
        3,
        4,
        criterion="origin_time",
    )
    dist_expr = alifestd_find_pair_distance_polars(
        df,
        3,
        4,
        criterion=pl.col("origin_time"),
    )
    assert dist_str == dist_expr


def test_find_pair_distance_polars_expr_computed():
    df = _make_simple_phylogeny()
    dist = alifestd_find_pair_distance_polars(
        df,
        3,
        4,
        criterion=pl.col("origin_time") * 2,
    )
    dist_str = alifestd_find_pair_distance_polars(
        df,
        3,
        4,
        criterion="origin_time",
    )
    assert dist == pytest.approx(dist_str * 2)


def test_calc_distance_matrix_polars_expr_matches_string():
    df = _make_simple_phylogeny()
    mat_str = alifestd_calc_distance_matrix_polars(
        df,
        criterion="origin_time",
    )
    mat_expr = alifestd_calc_distance_matrix_polars(
        df,
        criterion=pl.col("origin_time"),
    )
    np.testing.assert_array_equal(mat_str, mat_expr)


def test_downsample_tips_canopy_polars_expr_matches_string():
    df = _make_simple_phylogeny()
    result_str = alifestd_downsample_tips_canopy_polars(
        df,
        n_downsample=2,
        criterion="origin_time",
        ignore_topological_sensitivity=True,
    )
    result_expr = alifestd_downsample_tips_canopy_polars(
        df,
        n_downsample=2,
        criterion=pl.col("origin_time"),
        ignore_topological_sensitivity=True,
    )
    assert result_str.shape == result_expr.shape
