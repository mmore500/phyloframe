import polars as pl

from phyloframe.legacy._alifestd_make_ancestor_id_col_polars import (
    alifestd_make_ancestor_id_col_polars,
)


def test_alifestd_make_ancestor_id_col_polars_basic():
    """Convert ancestor_list to ancestor_id."""
    ids = pl.Series("id", [0, 1, 2])
    ancestor_lists = pl.Series("ancestor_list", ["[none]", "[0]", "[1]"])

    result = alifestd_make_ancestor_id_col_polars(ids, ancestor_lists)

    assert result.to_list() == [0, 0, 1]


def test_alifestd_make_ancestor_id_col_polars_empty_bracket():
    """Handle [] as root ancestor token."""
    ids = pl.Series("id", [0, 1])
    ancestor_lists = pl.Series("ancestor_list", ["[]", "[0]"])

    result = alifestd_make_ancestor_id_col_polars(ids, ancestor_lists)

    assert result.to_list() == [0, 0]


def test_alifestd_make_ancestor_id_col_polars_case_insensitive():
    """Handle [None] and [NONE] case variants."""
    ids = pl.Series("id", [0, 1])
    ancestor_lists = pl.Series("ancestor_list", ["[None]", "[0]"])

    result = alifestd_make_ancestor_id_col_polars(ids, ancestor_lists)

    assert result.to_list() == [0, 0]
