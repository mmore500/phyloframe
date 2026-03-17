import polars as pl

from phyloframe._auxlib._min_scalar_type_polars import min_scalar_type_polars


def test_small_negative():
    assert min_scalar_type_polars(-1) == pl.Int8


def test_small_positive():
    assert min_scalar_type_polars(1) == pl.Int8


def test_zero():
    assert min_scalar_type_polars(0) == pl.Int8


def test_int8_boundary():
    assert min_scalar_type_polars(-128) == pl.Int8
    assert min_scalar_type_polars(127) == pl.Int8


def test_int16_range():
    assert min_scalar_type_polars(-129) == pl.Int16
    assert min_scalar_type_polars(128) == pl.Int16


def test_int32_range():
    assert min_scalar_type_polars(-40000) == pl.Int32


def test_int64_range():
    assert min_scalar_type_polars(-(2**40)) == pl.Int64
