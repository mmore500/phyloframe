import pandas as pd
import pandas.testing as pdt
import polars as pl
import polars.testing as pltest
import pytest

from phyloframe.legacy import (
    alifestd_make_empty,
)
from phyloframe.legacy import (
    alifestd_prefix_roots_polars as alifestd_prefix_roots_polars_,
)
from phyloframe.legacy import alifestd_prefix_roots as alifestd_prefix_roots_

from ._impl import enforce_dtype_stability_pandas

alifestd_prefix_roots = enforce_dtype_stability_pandas(alifestd_prefix_roots_)
alifestd_prefix_roots_polars = enforce_dtype_stability_pandas(
    alifestd_prefix_roots_polars_,
)


def _call_polars(df, **kwargs):
    """Shorthand that sets allow_id_reassign=True by default."""
    kwargs.setdefault("allow_id_reassign", True)
    return alifestd_prefix_roots_polars(df, **kwargs)


def test_empty_df():
    df = alifestd_make_empty()
    result = alifestd_prefix_roots(df)
    assert result.empty

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(df))


@pytest.mark.parametrize("mutate", [False, True])
def test_single_root_with_origin_time(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0],
            "origin_time": [10],
            "ancestor_id": [0],
            "ancestor_list": ["[none]"],
        }
    )
    original = df.copy()

    result = alifestd_prefix_roots(df, origin_time=5, mutate=mutate)
    if not mutate:
        pdt.assert_frame_equal(df, original)

    expected = pd.DataFrame(
        {
            "id": [0, 1],
            "origin_time": [10, 5],
            "ancestor_id": [1, 1],
            "ancestor_list": ["[1]", "[]"],
        }
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True), expected, check_like=True
    )

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original))
    df.drop(columns=["ancestor_list"], inplace=True)
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original), origin_time=5)


@pytest.mark.parametrize("mutate", [False, True])
def test_single_root_without_origin_time(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0],
            "origin_time": [10],
            "ancestor_id": [0],
            "ancestor_list": ["[none]"],
        }
    )
    original = df.copy()
    result = alifestd_prefix_roots(df, origin_time=None, mutate=mutate)
    if not mutate:
        pdt.assert_frame_equal(df, original)

    expected = pd.DataFrame(
        {
            "id": [0, 1],
            "origin_time": [10, 0],
            "ancestor_id": [1, 1],
            "ancestor_list": ["[1]", "[]"],
        }
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True), expected, check_like=True
    )

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original))
    df.drop(columns=["ancestor_list"], inplace=True)
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(
            pl.from_pandas(original), origin_time=None
        )


@pytest.mark.parametrize("mutate", [False, True])
def test_multiple_roots(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "origin_time": [20, 15, 10, 5],
            "ancestor_id": [0, 0, 2, 2],
            "ancestor_list": ["[none]", "[0]", "[none]", "[2]"],
        }
    )
    original = df.copy()
    result = alifestd_prefix_roots(df, origin_time=12, mutate=mutate)
    if not mutate:
        pdt.assert_frame_equal(df, original)

    expected = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "origin_time": [20, 15, 10, 5, 12],
            "ancestor_id": [4, 0, 2, 2, 4],
            "ancestor_list": ["[4]", "[0]", "[none]", "[2]", "[]"],
        }
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True), expected, check_like=True
    )

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original))
    df.drop(columns=["ancestor_list"], inplace=True)
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original), origin_time=12)


@pytest.mark.parametrize("mutate", [False, True])
def test_no_eligible_roots(mutate: bool):

    df = pd.DataFrame(
        {
            "id": [0, 1],
            "origin_time": [10, 5],
            "ancestor_id": [0, 0],
            "ancestor_list": ["[none]", "[0]"],
        }
    )
    original = df.copy()
    result = alifestd_prefix_roots(df, origin_time=15, mutate=mutate)
    if not mutate:
        pdt.assert_frame_equal(df, original)

    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        original.reset_index(drop=True),
        check_like=True,
    )

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original))
    df.drop(columns=["ancestor_list"], inplace=True)
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original), origin_time=15)


@pytest.mark.parametrize("mutate", [False, True])
def test_warn_on_origin_time_delta(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0],
            "origin_time": [10],
            "origin_time_delta": [5],
            "ancestor_id": [0],
            "ancestor_list": ["[none]"],
        }
    )
    original = df.copy()
    with pytest.warns(UserWarning):
        alifestd_prefix_roots(df, origin_time=5, mutate=mutate)
    if not mutate:
        pdt.assert_frame_equal(df, original)

    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original))
    df.drop(columns=["ancestor_list"], inplace=True)
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(pl.from_pandas(original), origin_time=5)


@pytest.mark.parametrize("mutate", [False, True])
def test_fast_path_single_root(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "origin_time": [10, 9, 8],
            "ancestor_id": [0, 0, 1],
            "is_root": [True, False, False],
        },
    )
    original = df.copy()
    result = alifestd_prefix_roots(
        df,
        allow_id_reassign=True,
        origin_time=5,
        mutate=mutate,
    )
    if not mutate:
        pdt.assert_frame_equal(df, original)

    expected = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "origin_time": [5, 10, 9, 8],
            "ancestor_id": [0, 0, 1, 2],
        }
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True), expected, check_like=True
    )

    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        alifestd_prefix_roots_polars(
            pl.from_pandas(original), allow_id_reassign=True, origin_time=5
        ).to_pandas(),
        check_dtype=False,
    )


@pytest.mark.parametrize("mutate", [False, True])
def test_fast_path_multiple_roots(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "origin_time": [15, 22, 29, 28, 8, 9, 10],
            "ancestor_id": [0, 0, 1, 1, 4, 4, 4],
        },
    )
    original = df.copy()
    result = alifestd_prefix_roots(
        df,
        allow_id_reassign=True,
        origin_time=10,
        mutate=mutate,
    )
    if not mutate:
        pdt.assert_frame_equal(df, original)

    expected = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6, 7],
            "origin_time": [10, 15, 22, 29, 28, 8, 9, 10],
            "ancestor_id": [0, 0, 1, 2, 2, 5, 5, 5],
        },
    )
    pdt.assert_frame_equal(
        result.reset_index(drop=True), expected, check_like=True
    )

    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        alifestd_prefix_roots_polars(
            pl.from_pandas(original), allow_id_reassign=True, origin_time=10
        ).to_pandas(),
        check_dtype=False,
    )


@pytest.mark.parametrize("mutate", [False, True])
def test_fast_path_no_eligible_roots(mutate: bool):
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "origin_time": [5, 4, 3],
            "ancestor_id": [0, 0, 1],
        }
    )
    original = df.copy()
    result = alifestd_prefix_roots(
        df,
        allow_id_reassign=True,
        origin_time=10,
        mutate=mutate,
    )
    if not mutate:
        pdt.assert_frame_equal(df, original)
    pdt.assert_frame_equal(result, original)

    pdt.assert_frame_equal(
        result.reset_index(drop=True),
        alifestd_prefix_roots_polars(
            pl.from_pandas(original), allow_id_reassign=True, origin_time=10
        ).to_pandas(),
        check_dtype=False,
    )


# --- polars-specific tests ---


def _cross_check(pd_df, polars_result, **kwargs):
    """Run pandas implementation on same data and compare to polars result."""
    kwargs.setdefault("allow_id_reassign", True)
    pandas_result = alifestd_prefix_roots(pd_df, **kwargs)
    pdt.assert_frame_equal(
        pandas_result.reset_index(drop=True),
        polars_result.to_pandas(),
        check_dtype=False,
        check_like=True,
    )


def test_polars_empty_dataframe():
    df = pl.DataFrame({"id": [], "ancestor_id": [], "origin_time": []},).cast(
        {
            "id": pl.Int64,
            "ancestor_id": pl.Int64,
            "origin_time": pl.Float64,
        }
    )
    with pytest.raises(NotImplementedError):
        _call_polars(df)


def test_polars_ancestor_list_present():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
            "ancestor_list": ["[none]"],
        },
    )
    with pytest.raises(NotImplementedError):
        _call_polars(df)


def test_polars_allow_id_reassign_false():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
        },
    )
    with pytest.raises(NotImplementedError):
        alifestd_prefix_roots_polars(df, allow_id_reassign=False)


def test_polars_noncontiguous_ids():
    df = pl.DataFrame(
        {
            "id": [0, 2, 5],
            "ancestor_id": [0, 0, 2],
            "origin_time": [10.0, 9.0, 8.0],
        },
    )
    with pytest.raises(NotImplementedError):
        _call_polars(df)


def test_polars_ids_not_starting_at_zero():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "ancestor_id": [1, 1, 2],
            "origin_time": [10.0, 9.0, 8.0],
        },
    )
    with pytest.raises(NotImplementedError):
        _call_polars(df)


def test_polars_origin_time_specified_but_column_missing():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
        },
    )
    with pytest.raises(ValueError, match="origin_time specified"):
        _call_polars(df, origin_time=5)


def test_polars_origin_time_delta_warns():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
            "origin_time_delta": [5.0],
        },
    )
    with pytest.warns(UserWarning, match="origin_time_delta"):
        _call_polars(df)


def test_polars_singleton_no_origin_time():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
        },
    )
    result = _call_polars(df)
    expected = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [0.0, 10.0],
        },
    )
    pltest.assert_frame_equal(result, expected, check_dtypes=False)
    _cross_check(df.to_pandas(), result)


def test_polars_singleton_with_origin_time_eligible():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
        },
    )
    result = _call_polars(df, origin_time=5)
    expected = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [5.0, 10.0],
        },
    )
    pltest.assert_frame_equal(result, expected, check_dtypes=False)
    _cross_check(df.to_pandas(), result, origin_time=5)


def test_polars_singleton_with_origin_time_not_eligible():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [5.0],
        },
    )
    result = _call_polars(df, origin_time=10)
    pltest.assert_frame_equal(result, df, check_dtypes=False)
    _cross_check(df.to_pandas(), result, origin_time=10)


def test_polars_singleton_with_origin_time_equal_boundary():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "origin_time": [10.0],
        },
    )
    result = _call_polars(df, origin_time=10)
    # origin_time == threshold is not > threshold, so not eligible
    pltest.assert_frame_equal(result, df, check_dtypes=False)
    _cross_check(df.to_pandas(), result, origin_time=10)


def test_polars_single_tree_no_eligible_roots():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [5.0, 4.0, 3.0],
        },
    )
    result = _call_polars(df, origin_time=10)
    pltest.assert_frame_equal(result, df, check_dtypes=False)
    _cross_check(df.to_pandas(), result, origin_time=10)


def test_polars_multiple_roots_none_eligible():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 2, 2],
            "origin_time": [5.0, 4.0, 3.0, 2.0],
        },
    )
    result = _call_polars(df, origin_time=10)
    pltest.assert_frame_equal(result, df, check_dtypes=False)
    _cross_check(df.to_pandas(), result, origin_time=10)


def test_polars_two_roots_all_eligible_no_origin_time():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 2, 2],
            "origin_time": [10.0, 9.0, 8.0, 7.0],
        },
    )
    result = _call_polars(df)
    assert len(result) == 6  # 4 original + 2 prepended
    assert result["id"].to_list() == [0, 1, 2, 3, 4, 5]
    assert result["ancestor_id"][0] == 0
    assert result["ancestor_id"][1] == 1
    assert result["ancestor_id"][2] == 0
    assert result["ancestor_id"][3] == 2
    assert result["ancestor_id"][4] == 1
    assert result["ancestor_id"][5] == 4
    assert result["origin_time"][0] == 0.0
    assert result["origin_time"][1] == 0.0
    _cross_check(df.to_pandas(), result)


def test_polars_two_roots_one_eligible():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "ancestor_id": [0, 0, 2, 2],
            "origin_time": [20.0, 15.0, 5.0, 3.0],
        },
    )
    result = _call_polars(df, origin_time=10)
    assert len(result) == 5  # 4 original + 1 prepended
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"][0] == 0
    assert result["ancestor_id"][1] == 0
    assert result["ancestor_id"][2] == 1
    assert result["ancestor_id"][3] == 3
    assert result["ancestor_id"][4] == 3
    _cross_check(df.to_pandas(), result, origin_time=10)


def test_polars_three_independent_roots():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 1, 2],
            "origin_time": [10.0, 20.0, 30.0],
        },
    )
    result = _call_polars(df)
    assert len(result) == 6
    for i in range(3):
        assert result["ancestor_id"][i] == i
        assert result["origin_time"][i] == 0.0
    for i in range(3):
        assert result["ancestor_id"][i + 3] == i
    _cross_check(df.to_pandas(), result)


def test_polars_extra_columns_filled_with_null():
    df = pl.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "origin_time": [10.0, 9.0],
            "taxon_label": ["root", "child"],
            "fitness": [1.0, 0.5],
        },
    )
    result = _call_polars(df)
    assert "taxon_label" in result.columns
    assert "fitness" in result.columns
    assert result["taxon_label"][0] is None
    assert result["fitness"][0] is None
    assert result["taxon_label"][1] == "root"
    assert result["taxon_label"][2] == "child"
    assert result["fitness"][1] == 1.0
    assert result["fitness"][2] == 0.5
    _cross_check(df.to_pandas(), result)


def test_polars_is_root_column_dropped():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [10.0, 9.0, 8.0],
            "is_root": [True, False, False],
        },
    )
    result = _call_polars(df, origin_time=5)
    assert "is_root" not in result.columns
    _cross_check(df.to_pandas(), result, origin_time=5)


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64])
def test_polars_id_dtype_preserved(dtype):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=dtype),
            "ancestor_id": pl.Series([0, 0, 1], dtype=dtype),
            "origin_time": [10.0, 9.0, 8.0],
        },
    )
    result = _call_polars(df, origin_time=5)
    assert result["id"].dtype == dtype
    assert result["ancestor_id"].dtype == dtype
    if dtype == pl.Int64:
        _cross_check(df.to_pandas(), result, origin_time=5)


def test_polars_input_not_mutated():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
            "origin_time": [10.0, 9.0, 8.0],
        },
    )
    original = df.clone()
    result = _call_polars(df, origin_time=5)
    pltest.assert_frame_equal(df, original)
    _cross_check(df.to_pandas(), result, origin_time=5)
