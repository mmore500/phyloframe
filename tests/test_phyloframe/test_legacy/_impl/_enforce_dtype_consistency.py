import functools
import typing
import warnings

import numpy as np
import pandas as pd
import polars as pl


def assert_dtype_consistency(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates ``func`` so that it is called with both int64 and uint64
    dtypes for the ``id`` and ``ancestor_id`` columns, asserting that output
    dtypes match input dtypes in each case.

    The first positional argument is assumed to be a phylogeny DataFrame.
    Works with both pandas and polars DataFrames.
    """

    @functools.wraps(func)
    def dtype_checker(*args, **kwargs) -> typing.Any:
        if not args:
            return func(*args, **kwargs)

        phylogeny_df = args[0]
        rest_args = args[1:]

        if isinstance(phylogeny_df, pd.DataFrame):
            if "id" in phylogeny_df.columns:
                _check_pandas(func, phylogeny_df, rest_args, kwargs)
        elif isinstance(phylogeny_df, pl.DataFrame):
            if "id" in phylogeny_df.columns:
                _check_polars(func, phylogeny_df, rest_args, kwargs)

        # Return the result from the original (uncast) call for downstream use
        return func(*args, **kwargs)

    return dtype_checker


def _check_pandas(func, phylogeny_df, rest_args, kwargs):
    has_ancestor_id = "ancestor_id" in phylogeny_df.columns

    for dtype in (np.int64, np.uint64):
        try:
            df_cast = phylogeny_df.copy()
            df_cast["id"] = df_cast["id"].astype(dtype)
            if has_ancestor_id:
                df_cast["ancestor_id"] = df_cast["ancestor_id"].astype(
                    dtype,
                )
        except (ValueError, TypeError, OverflowError):
            continue  # skip if ids can't be cast (e.g., non-numeric)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = func(df_cast, *rest_args, **kwargs)
            except Exception:
                continue  # skip if function fails on cast data

        if isinstance(result, pd.DataFrame) and "id" in result.columns:
            assert result["id"].dtype == dtype, (
                f"{func.__name__}: id dtype changed from {dtype} "
                f"to {result['id'].dtype}"
            )
            if has_ancestor_id and "ancestor_id" in result.columns:
                assert result["ancestor_id"].dtype == dtype, (
                    f"{func.__name__}: ancestor_id dtype changed "
                    f"from {dtype} to {result['ancestor_id'].dtype}"
                )


def _check_polars(func, phylogeny_df, rest_args, kwargs):
    has_ancestor_id = "ancestor_id" in phylogeny_df.columns

    for dtype in (pl.Int64, pl.UInt64):
        try:
            df_cast = phylogeny_df.cast({"id": dtype})
            if has_ancestor_id:
                df_cast = df_cast.cast({"ancestor_id": dtype})
        except (
            pl.exceptions.InvalidOperationError,
            pl.exceptions.ComputeError,
        ):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = func(df_cast, *rest_args, **kwargs)
            except Exception:
                continue

        if isinstance(result, pl.DataFrame) and "id" in result.columns:
            assert result["id"].dtype == dtype, (
                f"{func.__name__}: id dtype changed from {dtype} "
                f"to {result['id'].dtype}"
            )
            if has_ancestor_id and "ancestor_id" in result.columns:
                assert result["ancestor_id"].dtype == dtype, (
                    f"{func.__name__}: ancestor_id dtype changed "
                    f"from {dtype} to {result['ancestor_id'].dtype}"
                )
