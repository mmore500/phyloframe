import typing

import numpy as np
import polars as pl


def calc_tiebroken_ranks(
    coordinates: typing.Sequence[np.ndarray],
) -> np.ndarray:
    """Rank items by each coordinate, breaking ties by the other
    coordinates rather than by item order.

    So, ranks do not depend on item order unless items tie on every
    coordinate. (`scipy.stats.rankdata`, by contrast, breaks ties by
    averaging or by item order.)

    Parameters
    ----------
    coordinates : Sequence[np.ndarray]
        Arrays of equal length, one per coordinate.

    Returns
    -------
    np.ndarray
        Integer array of shape (num_items, num_coordinates), with each
        item's rank by each coordinate, from 1 for the lowest to num_items
        for the highest.
    """
    df = pl.DataFrame(coordinates, orient="col")
    index = pl.int_range(pl.len())  # breaks full ties by item order
    return df.select(
        pl.struct(pl.col(c).alias("key"), *df.columns, index)
        .rank("ordinal")
        .alias(c)
        for c in df.columns
    ).to_numpy()
