import functools
import typing

import pandas as pd


def preserve_id_dtypes(func: typing.Callable) -> typing.Callable:
    """Decorator that preserves id/ancestor_id dtypes through a transform.

    Captures the dtypes of `id` and `ancestor_id` columns from the first
    positional argument (assumed to be a phylogeny DataFrame) before calling
    the wrapped function, then casts those columns back to the original dtypes
    on the returned DataFrame.

    Works with pandas DataFrames. Polars DataFrames are passed through
    unchanged (polars implementations should handle their own dtype
    consistency).
    """

    @functools.wraps(func)
    def wrapper(phylogeny_df, *args, **kwargs):
        if not isinstance(phylogeny_df, pd.DataFrame):
            return func(phylogeny_df, *args, **kwargs)

        id_dtype = phylogeny_df["id"].dtype
        has_ancestor_id = "ancestor_id" in phylogeny_df.columns
        ancestor_id_dtype = (
            phylogeny_df["ancestor_id"].dtype if has_ancestor_id else None
        )

        result = func(phylogeny_df, *args, **kwargs)

        if not isinstance(result, pd.DataFrame):
            return result

        if "id" in result.columns and result["id"].dtype != id_dtype:
            result["id"] = result["id"].astype(id_dtype)
        if (
            has_ancestor_id
            and "ancestor_id" in result.columns
            and result["ancestor_id"].dtype != ancestor_id_dtype
        ):
            result["ancestor_id"] = result["ancestor_id"].astype(
                ancestor_id_dtype,
            )

        return result

    return wrapper
