import polars as pl
import pytest

from phyloframe.legacy._alifestd_make_balanced_bifurcating_polars import (
    alifestd_make_balanced_bifurcating_polars,
)


def test_make_balanced_bifurcating_polars_depth0():
    result = alifestd_make_balanced_bifurcating_polars(0)
    assert result.is_empty()
    assert "id" in result.columns
    assert "ancestor_id" in result.columns


def test_make_balanced_bifurcating_polars_depth1():
    result = alifestd_make_balanced_bifurcating_polars(1)
    assert len(result) == 1
    assert result["id"].to_list() == [0]
    assert result["ancestor_id"].to_list() == [0]


def test_make_balanced_bifurcating_polars_depth2():
    result = alifestd_make_balanced_bifurcating_polars(2)
    assert len(result) == 3
    assert result["id"].to_list() == [0, 1, 2]
    assert result["ancestor_id"].to_list() == [0, 0, 0]


def test_make_balanced_bifurcating_polars_depth3():
    result = alifestd_make_balanced_bifurcating_polars(3)
    assert len(result) == 7


def test_make_balanced_bifurcating_polars_negative():
    with pytest.raises(ValueError):
        alifestd_make_balanced_bifurcating_polars(-1)
