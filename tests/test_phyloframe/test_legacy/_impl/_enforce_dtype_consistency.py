import functools
import typing

import pandas as pd
import polars as pl


def _check_id_dtypes(
    func_name: str,
    input_id_dtype: typing.Any,
    input_ancestor_id_dtype: typing.Optional[typing.Any],
    result: typing.Any,
    result_type: type,
) -> None:
    if not isinstance(result, result_type):
        return
    if "id" not in result.columns:
        return

    assert result["id"].dtype == input_id_dtype, (
        f"{func_name}: id dtype changed from {input_id_dtype} "
        f"to {result['id'].dtype}"
    )

    if input_ancestor_id_dtype is not None and "ancestor_id" in result.columns:
        assert result["ancestor_id"].dtype == input_ancestor_id_dtype, (
            f"{func_name}: ancestor_id dtype changed "
            f"from {input_ancestor_id_dtype} "
            f"to {result['ancestor_id'].dtype}"
        )


def enforce_dtype_stability_pandas(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates a pandas function to assert that output ``id`` and
    ``ancestor_id`` dtypes match the input dtypes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> typing.Any:
        if not args:
            return func(*args, **kwargs)

        phylogeny_df, *rest_args = args

        if not isinstance(phylogeny_df, pd.DataFrame):
            return func(phylogeny_df, *rest_args, **kwargs)
        if "id" not in phylogeny_df.columns:
            return func(phylogeny_df, *rest_args, **kwargs)

        id_dtype = phylogeny_df["id"].dtype
        ancestor_id_dtype = (
            phylogeny_df["ancestor_id"].dtype
            if "ancestor_id" in phylogeny_df.columns
            else None
        )

        result = func(phylogeny_df, *rest_args, **kwargs)

        _check_id_dtypes(
            func.__name__,
            id_dtype,
            ancestor_id_dtype,
            result,
            pd.DataFrame,
        )
        return result

    return wrapper


def enforce_dtype_stability_polars(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates a polars function to assert that output ``id`` and
    ``ancestor_id`` dtypes match the input dtypes."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> typing.Any:
        if not args:
            return func(*args, **kwargs)

        phylogeny_df, *rest_args = args

        if not isinstance(phylogeny_df, pl.DataFrame):
            return func(phylogeny_df, *rest_args, **kwargs)
        if "id" not in phylogeny_df.columns:
            return func(phylogeny_df, *rest_args, **kwargs)

        id_dtype = phylogeny_df["id"].dtype
        ancestor_id_dtype = (
            phylogeny_df["ancestor_id"].dtype
            if "ancestor_id" in phylogeny_df.columns
            else None
        )

        result = func(phylogeny_df, *rest_args, **kwargs)

        _check_id_dtypes(
            func.__name__,
            id_dtype,
            ancestor_id_dtype,
            result,
            pl.DataFrame,
        )
        return result

    return wrapper


def enforce_dtype_stability(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates ``func`` to assert that output ``id`` and ``ancestor_id``
    dtypes match the input dtypes.

    Works with both pandas and polars DataFrames. Prefer using the
    framework-specific variants ``enforce_dtype_stability_pandas`` or
    ``enforce_dtype_stability_polars`` when the input type is known.
    """
    return enforce_dtype_stability_pandas(
        enforce_dtype_stability_polars(func),
    )
