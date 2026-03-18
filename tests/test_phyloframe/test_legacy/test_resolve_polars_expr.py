import numpy as np
import polars as pl
import pytest

from phyloframe.legacy._alifestd_calc_distance_matrix_polars import (
    alifestd_calc_distance_matrix_polars,
)
from phyloframe.legacy._alifestd_coarsen_dilate_polars import (
    alifestd_coarsen_dilate_polars,
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


class TestSortChildrenPolarsExpr:
    def test_expr_matches_string(self):
        df = _make_simple_phylogeny()
        result_str = alifestd_sort_children_polars(df, "origin_time")
        result_expr = alifestd_sort_children_polars(
            df,
            pl.col("origin_time"),
        )
        assert result_str.equals(result_expr)

    def test_expr_computed(self):
        df = _make_simple_phylogeny()
        result = alifestd_sort_children_polars(
            df,
            pl.col("origin_time") * -1,
            reverse=True,
        )
        assert result.shape == df.shape
        # Temp column should be cleaned up
        assert all(not c.startswith("__phyloframe") for c in result.columns)


class TestMarkSampleTipsCanopyPolarsExpr:
    def test_expr_matches_string(self):
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

    def test_expr_computed(self):
        df = _make_simple_phylogeny()
        result = alifestd_mark_sample_tips_canopy_polars(
            df,
            n_sample=2,
            criterion=pl.col("origin_time") * 10,
        )
        assert "alifestd_mark_sample_tips_canopy_polars" in result.columns
        assert all(not c.startswith("__phyloframe") for c in result.columns)


class TestFindPairDistancePolarsExpr:
    def test_expr_matches_string(self):
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

    def test_expr_computed(self):
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


class TestCalcDistanceMatrixPolarsExpr:
    def test_expr_matches_string(self):
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


class TestCoarsenDilatePolarsExpr:
    def test_expr_matches_string(self):
        df = _make_simple_phylogeny()
        result_str = alifestd_coarsen_dilate_polars(
            df,
            criterion="origin_time",
            dilation=2,
            ignore_topological_sensitivity=True,
        )
        result_expr = alifestd_coarsen_dilate_polars(
            df,
            criterion=pl.col("origin_time"),
            dilation=2,
            ignore_topological_sensitivity=True,
        )
        assert result_str.shape == result_expr.shape
        assert all(
            not c.startswith("__phyloframe") for c in result_expr.columns
        )


class TestDownsampleTipsCanopyPolarsExpr:
    def test_expr_matches_string(self):
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
