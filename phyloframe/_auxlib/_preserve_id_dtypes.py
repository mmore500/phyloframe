import functools
import typing

import pandas as pd


def preserve_id_dtypes(func: typing.Callable) -> typing.Callable:
    """Decorator that preserves id/ancestor_id dtypes through a transform.

    Captures the dtypes of ``id`` and ``ancestor_id`` columns from the first
    positional argument (assumed to be a pandas phylogeny DataFrame) before
    calling the wrapped function, then casts those columns back to the
    original dtypes on the returned DataFrame.
    """

    @functools.wraps(func)
    def wrapper(phylogeny_df: pd.DataFrame, *args, **kwargs):
        id_dtype = phylogeny_df["id"].dtype
        ancestor_id_dtype = (
            phylogeny_df["ancestor_id"].dtype
            if "ancestor_id" in phylogeny_df.columns
            else id_dtype
        )

        result = func(phylogeny_df, *args, **kwargs)

        if "id" in result.columns:
            result["id"] = result["id"].astype(id_dtype)
        if "ancestor_id" in result.columns:
            result["ancestor_id"] = result["ancestor_id"].astype(
                ancestor_id_dtype,
            )

        return result

    return wrapper
