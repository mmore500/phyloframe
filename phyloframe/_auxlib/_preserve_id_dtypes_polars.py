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
    def wrapper(phylogeny_df, *args, **kwargs):
        if not isinstance(phylogeny_df, pl.DataFrame):
            return func(phylogeny_df, *args, **kwargs)

        schema = phylogeny_df.collect_schema()
        id_dtype = schema["id"]
        ancestor_id_dtype = (
            schema["ancestor_id"]
            if "ancestor_id" in schema.names()
            else id_dtype
        )

        result = func(phylogeny_df, *args, **kwargs)

        if not isinstance(result, pl.DataFrame):
            return result

        cast_map = {}
        if "id" in result.columns and result["id"].dtype != id_dtype:
            cast_map["id"] = id_dtype
        if (
            "ancestor_id" in result.columns
            and result["ancestor_id"].dtype != ancestor_id_dtype
        ):
            cast_map["ancestor_id"] = ancestor_id_dtype

        if cast_map:
            result = result.cast(cast_map)

        return result

    return wrapper
