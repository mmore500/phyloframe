import pytest

from phyloframe.legacy._alifestd_make_star_polars import (
    alifestd_make_star_polars,
)


def test_make_star_polars_0_leaves():
    result = alifestd_make_star_polars(0)
    assert result.is_empty()


def test_make_star_polars_1_leaf():
    result = alifestd_make_star_polars(1)
    assert len(result) == 1
    assert result["id"].to_list() == [0]
    assert result["ancestor_id"].to_list() == [0]


def test_make_star_polars_2_leaves():
    result = alifestd_make_star_polars(2)
    assert len(result) == 3
    assert result["id"].to_list() == [0, 1, 2]
    assert result["ancestor_id"].to_list() == [0, 0, 0]


def test_make_star_polars_4_leaves():
    result = alifestd_make_star_polars(4)
    assert len(result) == 5
    assert result["id"].to_list() == [0, 1, 2, 3, 4]
    assert result["ancestor_id"].to_list() == [0, 0, 0, 0, 0]


def test_make_star_polars_negative():
    with pytest.raises(ValueError):
        alifestd_make_star_polars(-1)
