import typing

import pandas as pd
import polars as pl

_supported_iterables = tuple, set, list, frozenset
_supported_mappings = dict
_pd_str_dtype = pd.Series(["_"]).dtype  # pandas 2/3 compat


def coerce_to_pandas(obj: typing.Any, *, recurse: bool = False) -> typing.Any:
    """
    If a Polars type is detected, coerce it to corresponding Pandas type.
    """
    if isinstance(obj, pl.LazyFrame):
        obj = obj.collect()

    if hasattr(obj, "__dataframe__"):
        return pd.api.interchange.from_dataframe(obj, allow_copy=True)
    elif isinstance(obj, pl.Series):
        result = obj.to_pandas()
        if obj.dtype == pl.Utf8 and result.dtype == object:
            result = result.astype(_pd_str_dtype)
        return result
    elif hasattr(obj, "to_pandas"):
        return obj.to_pandas()  # pyarrow is required for this operation
    elif recurse and isinstance(obj, _supported_iterables):
        return type(obj)(
            map(lambda x: coerce_to_pandas(x, recurse=recurse), obj)
        )
    elif recurse and isinstance(
        obj, _supported_mappings
    ):  # includes defaultdict etc
        return type(obj)(
            {k: coerce_to_pandas(v, recurse=recurse) for k, v in obj.items()},
        )
    else:
        return obj
