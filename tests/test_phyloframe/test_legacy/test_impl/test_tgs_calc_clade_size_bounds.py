import pytest

from phyloframe.legacy._impl import tgs_calc_clade_size_bounds


@pytest.mark.parametrize(
    "groups_per_doubling, expected",
    [
        (1, [2, 4, 8, 16, 32]),
        (2, [2, 3, 4, 6, 8, 12, 16, 24]),
        (4, [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20]),
    ],
)
def test_bounds(groups_per_doubling: int, expected: list):
    res = tgs_calc_clade_size_bounds(
        max_clade_size=17, groups_per_doubling=groups_per_doubling
    )
    assert res.tolist() == expected


def test_no_clades():
    assert tgs_calc_clade_size_bounds(
        max_clade_size=1, groups_per_doubling=2
    ).tolist() == [2]


def test_invalid():
    with pytest.raises(ValueError):
        tgs_calc_clade_size_bounds(max_clade_size=8, groups_per_doubling=0)
