import functools
import typing

import polars as pl


def enforce_dtype_stability_polars(
    func: typing.Callable,
) -> typing.Callable:
    """Decorates a polars function to assert that output ``id`` and
    ``ancestor_id`` dtypes match the input dtypes.

    Handles both ``pl.DataFrame`` and ``pl.LazyFrame`` inputs. Schema
    inspection is done lazily via ``collect_schema()`` to avoid
    materializing the full frame.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> typing.Any:
        if not args:
            return func(*args, **kwargs)

        phylogeny_df, *rest_args = args

        if not isinstance(phylogeny_df, (pl.DataFrame, pl.LazyFrame)):
            return func(phylogeny_df, *rest_args, **kwargs)

        input_schema = phylogeny_df.lazy().collect_schema()
        if "id" not in input_schema:
            return func(phylogeny_df, *rest_args, **kwargs)

        id_dtype = input_schema["id"]
        ancestor_id_dtype = (
            input_schema["ancestor_id"]
            if "ancestor_id" in input_schema
            else None
        )

        result = func(phylogeny_df, *rest_args, **kwargs)

        if isinstance(result, (pl.DataFrame, pl.LazyFrame)):
            result_schema = result.lazy().collect_schema()
            if "id" in result_schema:
                assert result_schema["id"] == id_dtype, (
                    f"{func.__name__}: id dtype changed "
                    f"from {id_dtype} to {result_schema['id']}"
                )
                if (
                    ancestor_id_dtype is not None
                    and "ancestor_id" in result_schema
                ):
                    assert result_schema["ancestor_id"] == ancestor_id_dtype, (
                        f"{func.__name__}: ancestor_id dtype changed "
                        f"from {ancestor_id_dtype} "
                        f"to {result_schema['ancestor_id']}"
                    )

        return result

    return wrapper
