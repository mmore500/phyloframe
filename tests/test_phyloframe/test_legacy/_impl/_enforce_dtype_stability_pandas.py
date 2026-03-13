import functools
import typing

import pandas as pd


def enforce_dtype_stability_pandas(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates a pandas function to assert that output ``id`` and
    ``ancestor_id`` dtypes match the input dtypes.

    The first positional argument must be a pandas DataFrame with an
    ``id`` column.
    """

    @functools.wraps(func)
    def wrapper(phylogeny_df: pd.DataFrame, *args, **kwargs) -> typing.Any:
        if not isinstance(phylogeny_df, pd.DataFrame):
            raise ValueError(
                f"enforce_dtype_stability_pandas: expected pandas "
                f"DataFrame, got {type(phylogeny_df)}"
            )

        id_dtype = phylogeny_df["id"].dtype
        ancestor_id_dtype = (
            phylogeny_df["ancestor_id"].dtype
            if "ancestor_id" in phylogeny_df.columns
            else None
        )

        result = func(phylogeny_df, *args, **kwargs)

        if isinstance(result, pd.DataFrame):
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
