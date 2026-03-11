import functools
import typing

import numpy as np
import pandas as pd


def enforce_dtype_consistency(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates ``func`` so that it is called with both int64 and uint64
    dtypes for the ``id`` and ``ancestor_id`` columns, asserting that output
    dtypes match input dtypes in each case.

    The first positional argument is assumed to be a phylogeny DataFrame.
    """

    @functools.wraps(func)
    def dtype_checker(*args, **kwargs) -> typing.Any:
        phylogeny_df = args[0]
        rest_args = args[1:]

        if not isinstance(phylogeny_df, pd.DataFrame):
            return func(*args, **kwargs)

        has_ancestor_id = "ancestor_id" in phylogeny_df.columns

        for dtype in (np.int64, np.uint64):
            df_cast = phylogeny_df.copy()
            df_cast["id"] = df_cast["id"].astype(dtype)
            if has_ancestor_id:
                df_cast["ancestor_id"] = df_cast["ancestor_id"].astype(dtype)

            result = func(df_cast, *rest_args, **kwargs)

            if isinstance(result, pd.DataFrame):
                assert result["id"].dtype == dtype, (
                    f"{func.__name__}: id dtype changed from {dtype} "
                    f"to {result['id'].dtype}"
                )
                if has_ancestor_id and "ancestor_id" in result.columns:
                    assert result["ancestor_id"].dtype == dtype, (
                        f"{func.__name__}: ancestor_id dtype changed "
                        f"from {dtype} to {result['ancestor_id'].dtype}"
                    )

        # Return the result from the original (uncast) call for downstream use
        return func(*args, **kwargs)

    return dtype_checker
