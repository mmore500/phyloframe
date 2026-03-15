import functools
import typing

import polars as pl


def preserve_id_dtypes_polars(func: typing.Callable) -> typing.Callable:
    """Decorator that preserves id/ancestor_id dtypes through a transform.

    Captures the dtypes of ``id`` and ``ancestor_id`` columns from the first
    positional argument (assumed to be a polars phylogeny DataFrame) before
    calling the wrapped function, then casts those columns back to the
    original dtypes on the returned DataFrame.
    """

    @functools.wraps(func)
    def wrapper(phylogeny_df: pl.DataFrame, *args, **kwargs):
        schema = phylogeny_df.collect_schema()
        id_dtype = schema["id"]
        ancestor_id_dtype = schema.get("ancestor_id", id_dtype)

        result = func(phylogeny_df, *args, **kwargs)

        result_names = result.collect_schema().names()
        cast_map = {
            col: dtype
            for col, dtype in [
                ("id", id_dtype),
                ("ancestor_id", ancestor_id_dtype),
            ]
            if col in result_names
        }
        return result.cast(cast_map)

    return wrapper
