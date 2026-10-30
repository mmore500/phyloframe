import numpy as np
import pytest

from phyloframe.legacy import (
    alifestd_mark_lineage_affine_asexual,
    alifestd_mark_lineage_cumsum_asexual,
    alifestd_mark_num_children_asexual,
    alifestd_mark_to_array,
)


def test_single_value():
    res = alifestd_mark_to_array(
        alifestd_mark_lineage_cumsum_asexual,
        np.array([0, 0, 1, 0]),
        np.array([1, 2, 3, 4]),
    )
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, [1, 3, 6, 5])


def test_multiple_values():
    res = alifestd_mark_to_array(
        alifestd_mark_lineage_affine_asexual,
        np.array([0, 0, 1]),
        np.array([9.0, 2.0, 3.0]),  # scale
        np.array([1.0, 1.0, 1.0]),  # offset
    )
    np.testing.assert_allclose(res, [1.0, 3.0, 10.0])


def test_no_values():
    res = alifestd_mark_to_array(
        alifestd_mark_num_children_asexual,
        np.array([0, 0, 0, 1]),
    )
    np.testing.assert_array_equal(res, [2, 1, 0, 0])


def test_kwargs_forwarded():
    res = alifestd_mark_to_array(
        alifestd_mark_lineage_cumsum_asexual,
        np.array([0, 0, 1]),
        np.array([1, 2, 3]),
        reverse=True,
    )
    np.testing.assert_array_equal(res, [6, 5, 3])


def test_empty():
    res = alifestd_mark_to_array(
        alifestd_mark_lineage_cumsum_asexual,
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    assert len(res) == 0


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        alifestd_mark_to_array(
            alifestd_mark_lineage_cumsum_asexual,
            np.array([0, 0]),
            np.array([1, 2, 3]),
        )
