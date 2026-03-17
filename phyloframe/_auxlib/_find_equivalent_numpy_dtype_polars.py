import numpy as np
import polars as pl


def find_equivalent_numpy_dtype_polars(
    pl_dtype: pl.datatypes.DataType,
) -> np.dtype:
    """Convert a polars dtype to the corresponding numpy dtype."""
    return pl.Series([], dtype=pl_dtype).to_numpy().dtype
