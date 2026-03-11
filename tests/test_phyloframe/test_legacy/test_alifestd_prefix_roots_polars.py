import polars as pl
import polars.testing as plt
import pytest

from phyloframe.legacy import alifestd_prefix_roots_polars


def _call(df, **kwargs):
    """Shorthand that sets allow_id_reassign=True by default."""
    kwargs.setdefault("allow_id_reassign", True)
    return alifestd_prefix_roots_polars(df, **kwargs)


class TestNotImplementedErrors:
    """Cases that are explicitly unsupported and raise NotImplementedError."""

    def test_empty_dataframe(self):
        df = pl.DataFrame(
            {"id": [], "ancestor_id": [], "origin_time": []},
        ).cast(
            {
                "id": pl.Int64,
                "ancestor_id": pl.Int64,
                "origin_time": pl.Float64,
            }
        )
        with pytest.raises(NotImplementedError):
            _call(df)

    def test_ancestor_list_present(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
                "ancestor_list": ["[none]"],
            },
        )
        with pytest.raises(NotImplementedError):
            _call(df)

    def test_allow_id_reassign_false(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
            },
        )
        with pytest.raises(NotImplementedError):
            alifestd_prefix_roots_polars(df, allow_id_reassign=False)

    def test_noncontiguous_ids(self):
        df = pl.DataFrame(
            {
                "id": [0, 2, 5],
                "ancestor_id": [0, 0, 2],
                "origin_time": [10.0, 9.0, 8.0],
            },
        )
        with pytest.raises(NotImplementedError):
            _call(df)

    def test_ids_not_starting_at_zero(self):
        df = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "ancestor_id": [1, 1, 2],
                "origin_time": [10.0, 9.0, 8.0],
            },
        )
        with pytest.raises(NotImplementedError):
            _call(df)


class TestValidationErrors:
    """Cases that raise ValueError due to invalid arguments."""

    def test_origin_time_specified_but_column_missing(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
            },
        )
        with pytest.raises(ValueError, match="origin_time specified"):
            _call(df, origin_time=5)


class TestWarnings:
    """Cases that emit warnings."""

    def test_origin_time_delta_warns(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
                "origin_time_delta": [5.0],
            },
        )
        with pytest.warns(UserWarning, match="origin_time_delta"):
            _call(df)


class TestSingleton:
    """Single-node phylogeny cases."""

    def test_singleton_no_origin_time(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
            },
        )
        result = _call(df)
        expected = pl.DataFrame(
            {
                "id": [0, 1],
                "ancestor_id": [0, 0],
                "origin_time": [0.0, 10.0],
            },
        )
        plt.assert_frame_equal(result, expected, check_dtypes=False)

    def test_singleton_with_origin_time_eligible(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
            },
        )
        result = _call(df, origin_time=5)
        expected = pl.DataFrame(
            {
                "id": [0, 1],
                "ancestor_id": [0, 0],
                "origin_time": [5.0, 10.0],
            },
        )
        plt.assert_frame_equal(result, expected, check_dtypes=False)

    def test_singleton_with_origin_time_not_eligible(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [5.0],
            },
        )
        result = _call(df, origin_time=10)
        plt.assert_frame_equal(result, df, check_dtypes=False)

    def test_singleton_with_origin_time_equal_boundary(self):
        df = pl.DataFrame(
            {
                "id": [0],
                "ancestor_id": [0],
                "origin_time": [10.0],
            },
        )
        result = _call(df, origin_time=10)
        # origin_time == threshold is not > threshold, so not eligible
        plt.assert_frame_equal(result, df, check_dtypes=False)


class TestNoEligibleRoots:
    """Cases where no roots are eligible for prefixing."""

    def test_single_tree_origin_time_too_high(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [5.0, 4.0, 3.0],
            },
        )
        result = _call(df, origin_time=10)
        plt.assert_frame_equal(result, df, check_dtypes=False)

    def test_multiple_roots_none_eligible(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
                "origin_time": [5.0, 4.0, 3.0, 2.0],
            },
        )
        result = _call(df, origin_time=10)
        plt.assert_frame_equal(result, df, check_dtypes=False)


class TestMultipleRoots:
    """Cases with multiple roots in the phylogeny."""

    def test_two_roots_all_eligible_no_origin_time(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
                "origin_time": [10.0, 9.0, 8.0, 7.0],
            },
        )
        result = _call(df)
        assert len(result) == 6  # 4 original + 2 prepended
        # new roots have ids 0, 1; original nodes shifted to 2..5
        assert result["id"].to_list() == [0, 1, 2, 3, 4, 5]
        # new roots are self-referencing
        assert result["ancestor_id"][0] == 0
        assert result["ancestor_id"][1] == 1
        # original root 0 (now id=2) should point to new root 0
        assert result["ancestor_id"][2] == 0
        # original node 1 (now id=3) was child of root 0, ancestor shifted
        assert result["ancestor_id"][3] == 2
        # original root 2 (now id=4) should point to new root 1
        assert result["ancestor_id"][4] == 1
        # original node 3 (now id=5) was child of root 2, ancestor shifted
        assert result["ancestor_id"][5] == 4
        # new root origin times default to 0
        assert result["origin_time"][0] == 0.0
        assert result["origin_time"][1] == 0.0

    def test_two_roots_one_eligible(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2, 3],
                "ancestor_id": [0, 0, 2, 2],
                "origin_time": [20.0, 15.0, 5.0, 3.0],
            },
        )
        result = _call(df, origin_time=10)
        assert len(result) == 5  # 4 original + 1 prepended
        assert result["id"].to_list() == [0, 1, 2, 3, 4]
        # new root is self-referencing
        assert result["ancestor_id"][0] == 0
        # original root 0 (now id=1, origin_time=20 > 10) points to new root
        assert result["ancestor_id"][1] == 0
        # original node 1 (now id=2) child of root 0, ancestor shifted
        assert result["ancestor_id"][2] == 1
        # original root 2 (now id=3, origin_time=5 <= 10) stays self-ref
        assert result["ancestor_id"][3] == 3
        # original node 3 (now id=4) child of root 2, ancestor shifted
        assert result["ancestor_id"][4] == 3

    def test_three_independent_roots(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 1, 2],
                "origin_time": [10.0, 20.0, 30.0],
            },
        )
        result = _call(df)
        assert len(result) == 6  # 3 original + 3 prepended
        # all new roots should be self-referencing
        for i in range(3):
            assert result["ancestor_id"][i] == i
            assert result["origin_time"][i] == 0.0
        # each original root should point to its corresponding new root
        for i in range(3):
            assert result["ancestor_id"][i + 3] == i


class TestExtraColumns:
    """Cases testing that extra columns are preserved correctly."""

    def test_extra_columns_filled_with_null(self):
        df = pl.DataFrame(
            {
                "id": [0, 1],
                "ancestor_id": [0, 0],
                "origin_time": [10.0, 9.0],
                "taxon_label": ["root", "child"],
                "fitness": [1.0, 0.5],
            },
        )
        result = _call(df)
        assert "taxon_label" in result.columns
        assert "fitness" in result.columns
        # prepended root row should have null for extra columns
        assert result["taxon_label"][0] is None
        assert result["fitness"][0] is None
        # original values should be preserved
        assert result["taxon_label"][1] == "root"
        assert result["taxon_label"][2] == "child"
        assert result["fitness"][1] == 1.0
        assert result["fitness"][2] == 0.5


class TestIsRootColumn:
    """Cases testing that is_root column is dropped."""

    def test_is_root_column_dropped(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [10.0, 9.0, 8.0],
                "is_root": [True, False, False],
            },
        )
        result = _call(df, origin_time=5)
        assert "is_root" not in result.columns


class TestIdTypes:
    """Cases testing different id dtypes."""

    @pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64])
    def test_id_dtype_preserved(self, dtype):
        df = pl.DataFrame(
            {
                "id": pl.Series([0, 1, 2], dtype=dtype),
                "ancestor_id": pl.Series([0, 0, 1], dtype=dtype),
                "origin_time": [10.0, 9.0, 8.0],
            },
        )
        result = _call(df, origin_time=5)
        assert result["id"].dtype == dtype
        assert result["ancestor_id"].dtype == dtype


class TestInputNotMutated:
    """Cases verifying that the input dataframe is not mutated."""

    def test_input_unchanged(self):
        df = pl.DataFrame(
            {
                "id": [0, 1, 2],
                "ancestor_id": [0, 0, 1],
                "origin_time": [10.0, 9.0, 8.0],
            },
        )
        original = df.clone()
        _call(df, origin_time=5)
        plt.assert_frame_equal(df, original)
