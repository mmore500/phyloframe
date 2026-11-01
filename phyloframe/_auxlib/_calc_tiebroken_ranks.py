import typing

import numpy as np


def calc_tiebroken_ranks(
    coordinates: typing.Sequence[np.ndarray],
) -> np.ndarray:
    """Rank items by each coordinate, breaking ties by the other
    coordinates rather than by item order.

    So, ranks do not depend on item order unless items tie on every
    coordinate. Unlike `scipy.stats.rankdata`, which breaks ties by
    averaging or by item order.

    Parameters
    ----------
    coordinates : Sequence[np.ndarray]
        Arrays of equal length, one per coordinate.

    Returns
    -------
    np.ndarray
        Array of shape (num_items, num_coordinates), with each item's rank
        by each coordinate, scaled to (0, 1], i.e., from 1 / num_items for
        the lowest to 1 for the highest.
    """
    num_items = len(coordinates[0])
    ranks = np.empty((num_items, len(coordinates)))
    for i, x in enumerate(coordinates):
        others = [y for j, y in enumerate(coordinates) if j != i]
        # np.lexsort's last key is primary
        order = np.lexsort([*others[::-1], x])
        ranks[order, i] = np.arange(1, num_items + 1) / num_items
    return ranks
