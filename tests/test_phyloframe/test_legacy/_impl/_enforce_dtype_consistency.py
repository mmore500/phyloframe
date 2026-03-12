import functools
import typing

import pandas as pd
import polars as pl


def _check_pandas(
    func_name: str,
    phylogeny_df: pd.DataFrame,
    result: typing.Any,
) -> None:
    if not isinstance(result, pd.DataFrame):
        return
    if "id" not in result.columns:
        return

    id_dtype = phylogeny_df["id"].dtype
    assert result["id"].dtype == id_dtype, (
        f"{func_name}: id dtype changed from {id_dtype} "
        f"to {result['id'].dtype}"
    )

    if (
        "ancestor_id" in phylogeny_df.columns
        and "ancestor_id" in result.columns
    ):
        ancestor_id_dtype = phylogeny_df["ancestor_id"].dtype
        assert result["ancestor_id"].dtype == ancestor_id_dtype, (
            f"{func_name}: ancestor_id dtype changed "
            f"from {ancestor_id_dtype} to {result['ancestor_id'].dtype}"
        )


def _check_polars(
    func_name: str,
    phylogeny_df: pl.DataFrame,
    result: typing.Any,
) -> None:
    if not isinstance(result, pl.DataFrame):
        return
    if "id" not in result.columns:
        return

    id_dtype = phylogeny_df["id"].dtype
    assert result["id"].dtype == id_dtype, (
        f"{func_name}: id dtype changed from {id_dtype} "
        f"to {result['id'].dtype}"
    )

    if (
        "ancestor_id" in phylogeny_df.columns
        and "ancestor_id" in result.columns
    ):
        ancestor_id_dtype = phylogeny_df["ancestor_id"].dtype
        assert result["ancestor_id"].dtype == ancestor_id_dtype, (
            f"{func_name}: ancestor_id dtype changed "
            f"from {ancestor_id_dtype} to {result['ancestor_id'].dtype}"
        )


def enforce_dtype_stability(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates ``func`` to assert that output ``id`` and ``ancestor_id``
    dtypes match the input dtypes.

    The first positional argument is assumed to be a phylogeny DataFrame.
    Works with both pandas and polars DataFrames.
    """

    @functools.wraps(func)
    def dtype_checker(*args, **kwargs) -> typing.Any:
        if not args:
            return func(*args, **kwargs)

        phylogeny_df, *rest_args = args

        result = func(phylogeny_df, *rest_args, **kwargs)

        if (
            isinstance(phylogeny_df, pd.DataFrame)
            and "id" in phylogeny_df.columns
        ):
            _check_pandas(func.__name__, phylogeny_df, result)
        elif (
            isinstance(phylogeny_df, pl.DataFrame)
            and "id" in phylogeny_df.columns
        ):
            _check_polars(func.__name__, phylogeny_df, result)

        return result

    return dtype_checker


# backward compatibility alias
assert_dtype_consistency = enforce_dtype_stability
