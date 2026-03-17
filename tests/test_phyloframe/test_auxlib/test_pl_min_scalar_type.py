import polars as pl

from phyloframe._auxlib._pl_min_scalar_type import pl_min_scalar_type


def test_small_negative():
    assert pl_min_scalar_type(-1) == pl.Int8


def test_small_positive():
    assert pl_min_scalar_type(1) == pl.Int8


def test_zero():
    assert pl_min_scalar_type(0) == pl.Int8


def test_int8_boundary():
    assert pl_min_scalar_type(-128) == pl.Int8
    assert pl_min_scalar_type(127) == pl.Int8


def test_int16_range():
    assert pl_min_scalar_type(-129) == pl.Int16
    assert pl_min_scalar_type(128) == pl.Int16


def test_int32_range():
    assert pl_min_scalar_type(-40000) == pl.Int32


def test_int64_range():
    assert pl_min_scalar_type(-(2**40)) == pl.Int64
