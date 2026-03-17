import polars as pl


def min_scalar_type_polars(value: int) -> pl.datatypes.DataType:
    """Return the smallest signed integer polars dtype that can hold *value*.

    Analogous to ``numpy.min_scalar_type`` but returns a polars dtype.
    """
    return pl.Series([value]).shrink_dtype().dtype
