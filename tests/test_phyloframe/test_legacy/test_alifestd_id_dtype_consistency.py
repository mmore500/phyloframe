import pandas as pd
import polars as pl
import pytest

from phyloframe.legacy import (
    alifestd_add_inner_leaves,
    alifestd_assign_contiguous_ids,
    alifestd_prefix_roots,
    alifestd_splay_polytomies,
)
from phyloframe.legacy._alifestd_assign_contiguous_ids_polars import (
    alifestd_assign_contiguous_ids_polars,
)
from phyloframe.legacy._alifestd_prefix_roots_polars import (
    alifestd_prefix_roots_polars,
)
from phyloframe.legacy._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)

from ._impl import enforce_dtype_consistency

alifestd_assign_contiguous_ids_ = enforce_dtype_consistency(
    alifestd_assign_contiguous_ids,
)
alifestd_prefix_roots_ = enforce_dtype_consistency(alifestd_prefix_roots)
alifestd_splay_polytomies_ = enforce_dtype_consistency(
    alifestd_splay_polytomies,
)
alifestd_add_inner_leaves_ = enforce_dtype_consistency(
    alifestd_add_inner_leaves,
)
alifestd_assign_contiguous_ids_polars_ = enforce_dtype_consistency(
    alifestd_assign_contiguous_ids_polars,
)
alifestd_prefix_roots_polars_ = enforce_dtype_consistency(
    alifestd_prefix_roots_polars,
)
alifestd_try_add_ancestor_id_col_polars_ = enforce_dtype_consistency(
    alifestd_try_add_ancestor_id_col_polars,
)


# --- pandas: alifestd_assign_contiguous_ids ---


def test_assign_contiguous_ids_dtype_simple():
    df = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
            "ancestor_list": ["[none]", "[10]", "[20]"],
        }
    )
    alifestd_assign_contiguous_ids_(df)


def test_assign_contiguous_ids_dtype_single_node():
    df = pd.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
            "ancestor_list": ["[none]"],
        }
    )
    alifestd_assign_contiguous_ids_(df)


# --- pandas: alifestd_prefix_roots ---


def test_prefix_roots_dtype_slow_path():
    df = pd.DataFrame(
        {
            "id": [0],
            "origin_time": [10],
            "ancestor_id": [0],
            "ancestor_list": ["[none]"],
        }
    )
    alifestd_prefix_roots_(df, origin_time=5)


def test_prefix_roots_dtype_fast_path():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "origin_time": [10, 9, 8],
            "ancestor_id": [0, 0, 1],
        },
    )
    alifestd_prefix_roots_(df, allow_id_reassign=True, origin_time=5)


def test_prefix_roots_dtype_multiple_roots():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "origin_time": [20, 15, 10, 5],
            "ancestor_id": [0, 0, 2, 2],
            "ancestor_list": ["[none]", "[0]", "[none]", "[2]"],
        }
    )
    alifestd_prefix_roots_(df, origin_time=12)


# --- pandas: alifestd_splay_polytomies ---


def test_splay_polytomies_dtype_no_polytomies():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_list": ["[none]", "[0]", "[1]"],
            "ancestor_id": [0, 0, 1],
        }
    )
    alifestd_splay_polytomies_(df)


def test_splay_polytomies_dtype_with_polytomy():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5, 6],
            "ancestor_list": [
                "[1]",
                "[none]",
                "[1]",
                "[1]",
                "[2]",
                "[3]",
                "[5]",
            ],
            "ancestor_id": [1, 1, 1, 1, 2, 3, 5],
        }
    )
    alifestd_splay_polytomies_(df)


# --- pandas: alifestd_add_inner_leaves ---


def test_add_inner_leaves_dtype_chain():
    df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "ancestor_id": [0, 0, 1],
        },
    )
    alifestd_add_inner_leaves_(df)


def test_add_inner_leaves_dtype_with_ancestor_list():
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "ancestor_id": [0, 0],
            "ancestor_list": ["[none]", "[0]"],
        }
    )
    alifestd_add_inner_leaves_(df)


# --- polars: alifestd_assign_contiguous_ids_polars ---


def test_assign_contiguous_ids_polars_dtype_simple():
    df = pl.DataFrame(
        {
            "id": [10, 20, 30],
            "ancestor_id": [10, 10, 20],
        }
    )
    alifestd_assign_contiguous_ids_polars_(df)


def test_assign_contiguous_ids_polars_dtype_single_node():
    df = pl.DataFrame(
        {
            "id": [0],
            "ancestor_id": [0],
        }
    )
    alifestd_assign_contiguous_ids_polars_(df)


# --- polars: alifestd_prefix_roots_polars ---


def test_prefix_roots_polars_dtype():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2],
            "origin_time": [10, 9, 8],
            "ancestor_id": [0, 0, 1],
        },
    )
    alifestd_prefix_roots_polars_(
        df,
        allow_id_reassign=True,
        origin_time=5,
    )


def test_prefix_roots_polars_dtype_multiple_roots():
    df = pl.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "origin_time": [20, 15, 10, 5],
            "ancestor_id": [0, 0, 2, 2],
        },
    )
    alifestd_prefix_roots_polars_(
        df,
        allow_id_reassign=True,
        origin_time=12,
    )


# --- polars: alifestd_try_add_ancestor_id_col_polars ---


@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.UInt64])
def test_try_add_ancestor_id_col_polars_dtype(id_dtype):
    df = pl.DataFrame(
        {
            "id": pl.Series([0, 1, 2], dtype=id_dtype),
            "ancestor_list": ["[none]", "[0]", "[1]"],
        }
    )
    result = alifestd_try_add_ancestor_id_col_polars(df)
    assert result["ancestor_id"].dtype == id_dtype, (
        f"ancestor_id dtype {result['ancestor_id'].dtype} != id dtype "
        f"{id_dtype}"
    )
