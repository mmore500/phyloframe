import typing

import polars as pl

from .._auxlib._min_scalar_type_polars import min_scalar_type_polars
from ._alifestd_from_avida_spop import (
    _AVIDA_TO_ALIFE_FIELD,
    _parse_spop_ancestor_list,
    _parse_spop_text,
)


def alifestd_from_avida_spop_polars(
    spop_text: str,
    *,
    create_ancestor_list: bool = False,
    dtype_id: typing.Optional[pl.datatypes.DataType] = pl.Int64,
) -> pl.DataFrame:
    """Convert Avida ``.spop`` population snapshot text to a phylogeny
    dataframe.

    Parses the text content of an Avida ``.spop`` (structured population)
    file and returns a polars DataFrame in alife standard format.

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.
    dtype_id : pl.DataType or None, default pl.Int64
        Polars dtype for the ``id`` column. If None, the smallest signed
        integer dtype is chosen automatically based on the maximum id
        value in the data.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe in alife standard format.

    See Also
    --------
    alifestd_from_avida_spop :
        Pandas-based implementation.

    Raises
    ------
    ValueError
        If the ``#format`` header is missing from the spop text.
    """
    header, avida_data = _parse_spop_text(spop_text)

    if dtype_id is None:
        if avida_data["id"]:
            max_id = max(int(v) for v in avida_data["id"])
            pl_dtype_id = min_scalar_type_polars(-max(max_id, 1))
        else:
            pl_dtype_id = min_scalar_type_polars(-1)
    else:
        pl_dtype_id = dtype_id

    if not avida_data["id"]:
        columns = {"id": pl.Series([], dtype=pl_dtype_id)}
        if create_ancestor_list:
            columns["ancestor_list"] = pl.Series([], dtype=pl.Utf8)
        columns["origin_time"] = pl.Series([], dtype=pl.Int64)
        return pl.DataFrame(columns)

    # Build alife-standard columns.
    result_data = {}
    result_data["id"] = pl.Series(avida_data["id"]).cast(pl_dtype_id)

    if create_ancestor_list:
        result_data["ancestor_list"] = pl.Series(
            [_parse_spop_ancestor_list(p) for p in avida_data["parents"]],
            dtype=pl.Utf8,
        )

    result_data["origin_time"] = pl.Series(
        avida_data["update_born"],
    ).cast(pl.Int64)

    # Add remaining Avida fields with standard names.
    skip_avida = {"id", "parents", "update_born"}
    for avida_field, alife_field in _AVIDA_TO_ALIFE_FIELD.items():
        if (
            avida_field in avida_data
            and avida_field not in skip_avida
            and alife_field not in result_data
        ):
            result_data[alife_field] = pl.Series(
                avida_data[avida_field],
                dtype=pl.Utf8,
            )

    return pl.DataFrame(result_data)
