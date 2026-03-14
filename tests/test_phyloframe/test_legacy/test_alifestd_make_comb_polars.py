import pytest

from phyloframe.legacy._alifestd_make_comb_polars import (
    alifestd_make_comb_polars,
)


def test_make_comb_polars_0_leaves():
    result = alifestd_make_comb_polars(0)
    assert result.is_empty()


def test_make_comb_polars_1_leaf():
    result = alifestd_make_comb_polars(1)
    assert len(result) == 1
    assert result["id"].to_list() == [0]


def test_make_comb_polars_2_leaves():
    result = alifestd_make_comb_polars(2)
    assert len(result) == 3


def test_make_comb_polars_4_leaves():
    result = alifestd_make_comb_polars(4)
    assert len(result) == 7


def test_make_comb_polars_negative():
    with pytest.raises(ValueError):
        alifestd_make_comb_polars(-1)
