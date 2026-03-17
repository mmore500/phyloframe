import numpy as np
import polars as pl

from phyloframe._auxlib._pl_dtype_to_np_dtype import pl_dtype_to_np_dtype


def test_int8():
    assert pl_dtype_to_np_dtype(pl.Int8) == np.dtype(np.int8)


def test_int16():
    assert pl_dtype_to_np_dtype(pl.Int16) == np.dtype(np.int16)


def test_int32():
    assert pl_dtype_to_np_dtype(pl.Int32) == np.dtype(np.int32)


def test_int64():
    assert pl_dtype_to_np_dtype(pl.Int64) == np.dtype(np.int64)


def test_float32():
    assert pl_dtype_to_np_dtype(pl.Float32) == np.dtype(np.float32)


def test_float64():
    assert pl_dtype_to_np_dtype(pl.Float64) == np.dtype(np.float64)
