import numpy as np
import pytest

from phyloframe._auxlib import calc_geometric_bins


def test_ratio_two():
    values = np.arange(1, 18)
    expected = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5]
    assert calc_geometric_bins(values, ratio=2.0).tolist() == expected


@pytest.mark.parametrize("ratio", [1.5, 3.0, 10.0])
def test_exact_powers_close_their_bin(ratio: float):
    powers = ratio ** np.arange(8)
    assert calc_geometric_bins(powers, ratio=ratio).tolist() == list(range(8))
    above = powers * (1 + 1e-6)
    assert calc_geometric_bins(above, ratio=ratio).tolist() == list(
        range(1, 9)
    )


def test_empty():
    assert len(calc_geometric_bins(np.array([]), ratio=2.0)) == 0
