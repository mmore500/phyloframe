import typing

import numpy as np
import pandas as pd

# Avida SPOP fields that contain comma-delimited sets.
_AVIDA_SET_FIELDS = frozenset({"parents", "cells", "gest_offset", "lineage"})

# Mapping from Avida field names to alife standard column names.
_AVIDA_TO_ALIFE_FIELD = {
    "id": "id",
    "parents": "ancestor_list",
    "update_born": "origin_time",
    "src": "src",
    "src_args": "src_args",
    "num_units": "num_units",
    "total_units": "total_units",
    "length": "length",
    "merit": "merit",
    "gest_time": "gest_time",
    "fitness": "fitness",
    "gen_born": "gen_born",
    "update_deactivated": "update_deactivated",
    "depth": "depth",
    "hw_type": "hw_type",
    "inst_set": "inst_set",
    "sequence": "sequence",
    "cells": "cells",
    "gest_offset": "gest_offset",
    "lineage": "lineage",
}


def _parse_spop_header(line: str) -> typing.List[str]:
    """Extract field names from the ``#format`` header line.

    Parameters
    ----------
    line : str
        A line beginning with ``#format``.

    Returns
    -------
    list of str
        Field names in order.
    """
    return line.replace("#format", "").strip().split()


def _parse_spop_ancestor_list(
    raw_parents: str,
) -> str:
    """Convert Avida parents field to alife-standard ancestor_list string.

    Parameters
    ----------
    raw_parents : str
        Comma-delimited parent IDs, or ``"(none)"`` for the root.

    Returns
    -------
    str
        Alife-standard ancestor_list value, e.g. ``"[none]"`` or
        ``"[123,456]"``.
    """
    if raw_parents == "(none)":
        return "[none]"
    return "[" + raw_parents + "]"


def alifestd_from_avida_spop(
    spop_text: str,
    *,
    create_ancestor_list: bool = True,
) -> pd.DataFrame:
    """Convert Avida ``.spop`` population snapshot text to a phylogeny
    dataframe.

    Parses the text content of an Avida ``.spop`` (structured population)
    file and returns a pandas DataFrame in alife standard format.

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.
    create_ancestor_list : bool, default True
        If True, include an ``ancestor_list`` column in the result.

    Returns
    -------
    pd.DataFrame
        Phylogeny dataframe in alife standard format.

    See Also
    --------
    alifestd_from_avida_spop_polars :
        Polars-based implementation.

    Raises
    ------
    ValueError
        If the ``#format`` header is missing from the spop text.
    """
    header = None
    data_lines = []

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
        columns = {"id": pd.Series(dtype=np.int64)}
        if create_ancestor_list:
            columns["ancestor_list"] = pd.Series(dtype=str)
        columns["origin_time"] = pd.Series(dtype=np.int64)
        return pd.DataFrame(columns)

    # Parse data rows.
    avida_data: typing.Dict[str, typing.List] = {field: [] for field in header}
    for line in data_lines:
        parts = line.split(" ")
        for i, field in enumerate(header):
            value = parts[i] if i < len(parts) else "NONE"
            avida_data[field].append(value)

    # Build alife-standard columns.
    result_data: typing.Dict[str, typing.Any] = {}
    result_data["id"] = pd.array(avida_data["id"], dtype=np.int64)

    if create_ancestor_list:
        result_data["ancestor_list"] = [
            _parse_spop_ancestor_list(p) for p in avida_data["parents"]
        ]

    result_data["origin_time"] = pd.array(
        avida_data["update_born"],
        dtype=np.int64,
    )

    # Add remaining Avida fields with standard names.
    skip_avida = {"id", "parents", "update_born"}
    for avida_field, alife_field in _AVIDA_TO_ALIFE_FIELD.items():
        if (
            avida_field in avida_data
            and avida_field not in skip_avida
            and alife_field not in result_data
        ):
            result_data[alife_field] = avida_data[avida_field]

    return pd.DataFrame(result_data)
