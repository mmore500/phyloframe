import functools
import inspect
import typing

import polars as pl


def _resolve_polars_expr(
    *kwarg_names: str,
) -> typing.Callable:
    """Decorator that resolves kwargs from ``pl.Expr`` to column names.

    For each named keyword argument, if the caller passes a
    ``polars.Expr`` instead of a column-name string the decorator
    materializes the expression as a temporary column in
    ``phylogeny_df`` (the first parameter of the wrapped function) and
    replaces the argument with that column's name.  If the wrapped
    function returns a ``polars.DataFrame``, any temporary columns are
    dropped before the result is returned to the caller.

    Parameters
    ----------
    *kwarg_names : str
        Names of keyword arguments to resolve (e.g.
        ``"criterion"``, ``"criterion_delta"``).

    Examples
    --------
    >>> @_resolve_polars_expr("criterion")
    ... def my_func(df, *, criterion="origin_time"):
    ...     return df.select(criterion)
    """

    def decorator(
        func: typing.Callable,
    ) -> typing.Callable:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            phylogeny_df = bound.arguments["phylogeny_df"]
            temp_cols = []

            for name in kwarg_names:
                value = bound.arguments[name]
                if isinstance(value, str):
                    continue
                if not isinstance(value, pl.Expr):
                    raise TypeError(
                        f"{name} must be a str or polars.Expr, "
                        f"got {type(value).__name__}",
                    )
                suffix = f"_{name}" if len(kwarg_names) > 1 else ""
                temp_name = f"__phyloframe_resolved_expr{suffix}__"
                phylogeny_df = phylogeny_df.with_columns(
                    value.alias(temp_name),
                )
                bound.arguments[name] = temp_name
                temp_cols.append(temp_name)

            bound.arguments["phylogeny_df"] = phylogeny_df
            result = func(*bound.args, **bound.kwargs)

            if temp_cols and isinstance(result, pl.DataFrame):
                existing = set(result.columns)
                to_drop = [c for c in temp_cols if c in existing]
                if to_drop:
                    result = result.drop(to_drop)

            return result

        return wrapper

    return decorator
