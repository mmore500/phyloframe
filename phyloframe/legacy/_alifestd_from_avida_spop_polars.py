import typing

import polars as pl

from ._alifestd_from_avida_spop import (
    _AVIDA_TO_ALIFE_FIELD,
    _parse_spop_ancestor_list,
    _parse_spop_header,
)

# Avida SPOP fields that contain comma-delimited sets.
_AVIDA_SET_FIELDS = frozenset({"parents", "cells", "gest_offset", "lineage"})


def alifestd_from_avida_spop_polars(
    spop_text: str,
    *,
    create_ancestor_list: bool = True,
) -> pl.DataFrame:
    """Convert Avida ``.spop`` population snapshot text to a phylogeny
    dataframe.

    Parses the text content of an Avida ``.spop`` (structured population)
    file and returns a polars DataFrame in alife standard format.

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.
    create_ancestor_list : bool, default True
        If True, include an ``ancestor_list`` column in the result.

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
    header = None
    data_lines: typing.List[str] = []

    for line in spop_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] == "#":
            if stripped.startswith("#format"):
                header = _parse_spop_header(stripped)
            continue
        data_lines.append(stripped)

    if header is None:
        raise ValueError(
            "Failed to find #format header in spop text.",
        )

    if not data_lines:
        columns = {"id": pl.Series([], dtype=pl.Int64)}
        if create_ancestor_list:
            columns["ancestor_list"] = pl.Series([], dtype=pl.Utf8)
        columns["origin_time"] = pl.Series([], dtype=pl.Int64)
        return pl.DataFrame(columns)

    # Parse data rows.
    avida_data: typing.Dict[str, typing.List[str]] = {
        field: [] for field in header
    }
    for line in data_lines:
        parts = line.split(" ")
        for i, field in enumerate(header):
            value = parts[i] if i < len(parts) else "NONE"
            avida_data[field].append(value)

    # Build alife-standard columns.
    result_data: typing.Dict[str, typing.Any] = {}
    result_data["id"] = pl.Series(avida_data["id"]).cast(pl.Int64)

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
