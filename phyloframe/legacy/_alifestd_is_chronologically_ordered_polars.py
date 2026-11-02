from numbers import Number
import typing

import polars as pl

from ._alifestd_find_chronological_inconsistency_polars import (
    alifestd_find_chronological_inconsistency_polars,
)


def alifestd_is_chronologically_ordered_polars(
    phylogeny_df: pl.DataFrame,
    *,
    epsilon: typing.Optional[Number] = None,
) -> bool:
    """Check if all taxa have origin times at least `epsilon` after their
    ancestor's origin time.

    Root taxa are not checked.
    """
    return (
        alifestd_find_chronological_inconsistency_polars(
            phylogeny_df, epsilon=epsilon
        )
        is None
    )
