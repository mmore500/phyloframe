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
    return line.replace("#format", "").strip().split()


def _parse_spop_ancestor_list(raw_parents: str) -> str:
    if raw_parents == "(none)":
        return "[none]"
    return "[" + raw_parents + "]"


def _parse_spop_text(
    spop_text: str,
) -> typing.Tuple[typing.List[str], typing.Dict[str, typing.List[str]]]:
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

    avida_data: typing.Dict[str, typing.List[str]] = {
        field: [] for field in header
    }
    for line in data_lines:
        parts = line.split(" ")
        for i, field in enumerate(header):
            value = parts[i] if i < len(parts) else "NONE"
            avida_data[field].append(value)

    return header, avida_data


def alifestd_from_avida_spop(
    spop_text: str,
    *,
    create_ancestor_list: bool = False,
    dtype_id: typing.Optional[type] = np.int64,
) -> pd.DataFrame:
    """Convert Avida ``.spop`` population snapshot text to a phylogeny
    dataframe.

    Parses the text content of an Avida ``.spop`` (structured population)
    file and returns a pandas DataFrame in alife standard format.

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.
    dtype_id : type or None, default np.int64
        Numpy dtype for the ``id`` column. If None, the smallest signed
        integer dtype is chosen automatically based on the maximum id
        value in the data.

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
    header, avida_data = _parse_spop_text(spop_text)

    if dtype_id is None:
        if avida_data["id"]:
            max_id = max(int(v) for v in avida_data["id"])
            resolved_dtype_id = np.min_scalar_type(-max(max_id, 1))
        else:
            resolved_dtype_id = np.min_scalar_type(-1)
    else:
        resolved_dtype_id = np.dtype(dtype_id)

    if not avida_data["id"]:
        columns = {"id": pd.Series(dtype=resolved_dtype_id)}
        if create_ancestor_list:
            columns["ancestor_list"] = pd.Series(dtype=str)
        columns["origin_time"] = pd.Series(dtype=np.int64)
        return pd.DataFrame(columns)

    # Build alife-standard columns.
    result_data: typing.Dict[str, typing.Any] = {}
    result_data["id"] = pd.array(avida_data["id"], dtype=resolved_dtype_id)

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
