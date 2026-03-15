import numpy as np
import pytest

from phyloframe._auxlib._jit_parse_float import jit_parse_float


def _chars(s):
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8)


def test_integer():
    c = _chars("42")
    assert jit_parse_float(c, 0, 2) == pytest.approx(42.0)


def test_decimal():
    c = _chars("3.14")
    assert jit_parse_float(c, 0, 4) == pytest.approx(3.14)


def test_negative():
    c = _chars("-2.5")
    assert jit_parse_float(c, 0, 4) == pytest.approx(-2.5)


def test_positive_sign():
    c = _chars("+7.0")
    assert jit_parse_float(c, 0, 4) == pytest.approx(7.0)


def test_leading_dot():
    c = _chars(".5")
    assert jit_parse_float(c, 0, 2) == pytest.approx(0.5)


def test_trailing_dot():
    c = _chars("5.")
    assert jit_parse_float(c, 0, 2) == pytest.approx(5.0)


def test_zero():
    c = _chars("0")
    assert jit_parse_float(c, 0, 1) == pytest.approx(0.0)


def test_zero_point_zero():
    c = _chars("0.0")
    assert jit_parse_float(c, 0, 3) == pytest.approx(0.0)


def test_scientific_lowercase():
    c = _chars("1.5e3")
    assert jit_parse_float(c, 0, 5) == pytest.approx(1500.0)


def test_scientific_uppercase():
    c = _chars("2.0E4")
    assert jit_parse_float(c, 0, 5) == pytest.approx(20000.0)


def test_scientific_negative_exp():
    c = _chars("2.5e-2")
    assert jit_parse_float(c, 0, 6) == pytest.approx(0.025)


def test_scientific_positive_exp():
    c = _chars("7e+2")
    assert jit_parse_float(c, 0, 4) == pytest.approx(700.0)


def test_negative_scientific():
    c = _chars("-1.23e4")
    assert jit_parse_float(c, 0, 7) == pytest.approx(-12300.0)


def test_high_precision():
    c = _chars("3.141592653589793")
    assert jit_parse_float(c, 0, 17) == pytest.approx(
        3.141592653589793, rel=1e-12
    )


def test_leading_whitespace():
    c = _chars("  42")
    assert jit_parse_float(c, 0, 4) == pytest.approx(42.0)


def test_empty_range():
    c = _chars("abc")
    assert np.isnan(jit_parse_float(c, 0, 0))


def test_whitespace_only():
    c = _chars("   ")
    assert np.isnan(jit_parse_float(c, 0, 3))


def test_subrange():
    c = _chars("xxx12.5yyy")
    assert jit_parse_float(c, 3, 7) == pytest.approx(12.5)


def test_large_integer():
    c = _chars("123456789")
    assert jit_parse_float(c, 0, 9) == pytest.approx(123456789.0)


def test_small_scientific():
    c = _chars("1e-10")
    assert jit_parse_float(c, 0, 5) == pytest.approx(1e-10)


def test_large_scientific():
    c = _chars("9.99e10")
    assert jit_parse_float(c, 0, 7) == pytest.approx(9.99e10)


def test_negative_zero():
    c = _chars("-0.0")
    assert jit_parse_float(c, 0, 4) == pytest.approx(0.0)
