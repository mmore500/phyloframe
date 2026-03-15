import polars as pl

from phyloframe.legacy._alifestd_make_empty_polars import (
    alifestd_make_empty_polars,
)


def test_alifestd_make_empty_polars_basic():
    """Create empty dataframe with ancestor_id by default."""
    result = alifestd_make_empty_polars()
    assert result.is_empty()
    assert "id" in result.columns
    assert "ancestor_id" in result.columns
    assert result.schema["id"] == pl.Int64
    assert result.schema["ancestor_id"] == pl.Int64


def test_alifestd_make_empty_polars_with_ancestor_id():
    """Create empty dataframe with ancestor_id."""
    result = alifestd_make_empty_polars(ancestor_id=True)
    assert result.is_empty()
    assert "id" in result.columns
    assert "ancestor_id" in result.columns
    assert result.schema["id"] == pl.Int64
    assert result.schema["ancestor_id"] == pl.Int64
