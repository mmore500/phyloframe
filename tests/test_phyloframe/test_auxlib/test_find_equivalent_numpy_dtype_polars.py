import numpy as np
import polars as pl

from phyloframe._auxlib._find_equivalent_numpy_dtype_polars import (
    find_equivalent_numpy_dtype_polars,
)


def test_int8():
    assert find_equivalent_numpy_dtype_polars(pl.Int8) == np.dtype(np.int8)


def test_int16():
    assert find_equivalent_numpy_dtype_polars(pl.Int16) == np.dtype(np.int16)


def test_int32():
    assert find_equivalent_numpy_dtype_polars(pl.Int32) == np.dtype(np.int32)


def test_int64():
    assert find_equivalent_numpy_dtype_polars(pl.Int64) == np.dtype(np.int64)


def test_float32():
    assert find_equivalent_numpy_dtype_polars(pl.Float32) == np.dtype(
        np.float32
    )


def test_float64():
    assert find_equivalent_numpy_dtype_polars(pl.Float64) == np.dtype(
        np.float64
    )
