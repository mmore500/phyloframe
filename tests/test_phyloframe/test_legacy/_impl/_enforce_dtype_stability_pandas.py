import functools
import typing

import pandas as pd


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

        if isinstance(result, pd.DataFrame) and "id" in result.columns:
            assert result["id"].dtype == id_dtype, (
                f"{func.__name__}: id dtype changed from {id_dtype} "
                f"to {result['id'].dtype}"
            )
            if (
                ancestor_id_dtype is not None
                and "ancestor_id" in result.columns
            ):
                assert result["ancestor_id"].dtype == ancestor_id_dtype, (
                    f"{func.__name__}: ancestor_id dtype changed "
                    f"from {ancestor_id_dtype} "
                    f"to {result['ancestor_id'].dtype}"
                )

        return result

    return wrapper
