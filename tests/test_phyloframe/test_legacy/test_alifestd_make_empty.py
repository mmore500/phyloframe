from phyloframe.legacy import alifestd_make_empty as alifestd_make_empty_
from phyloframe.legacy import alifestd_validate

from ._impl import assert_dtype_consistency

alifestd_make_empty = assert_dtype_consistency(alifestd_make_empty_)


def test_alifestd_make_empty():
    assert len(alifestd_make_empty()) == 0
    assert alifestd_validate(alifestd_make_empty())
    assert "id" in alifestd_make_empty()
    assert "ancestor_list" in alifestd_make_empty()


def test_alifestd_make_empty_ancestor_id():
    assert len(alifestd_make_empty(ancestor_id=True)) == 0
    assert alifestd_validate(alifestd_make_empty(ancestor_id=True))
    assert "id" in alifestd_make_empty(ancestor_id=True)
    assert "ancestor_list" in alifestd_make_empty(ancestor_id=True)
    assert "ancestor_id" in alifestd_make_empty(ancestor_id=True)
