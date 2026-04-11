import argparse
import logging
import os
import pathlib
import typing

import numpy as np
import pandas as pd

from .._auxlib._configure_prod_logging import configure_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration


def _parse_spop_header(line: str) -> typing.List[str]:
    """Extract field names from the ``#format`` header line."""
    return line.replace("#format", "").strip().split()


def _parse_spop_ancestor_list(raw_parents: str) -> str:
    """Convert Avida parents field to alife-standard ancestor_list string."""
    if raw_parents == "(none)":
        return "[none]"
    return "[" + raw_parents + "]"


def _parse_spop_text(
    spop_text: str,
) -> typing.Tuple[typing.List[str], typing.Dict[str, typing.List[str]]]:
    """Parse raw spop text into a header and per-field string lists.

    Implementation detail shared by ``alifestd_from_avida_spop`` and
    ``alifestd_from_avida_spop_polars``.

    Adapted from
    https://github.com/alife-data-standards/converters-avida

    Parameters
    ----------
    spop_text : str
        Full text content of an Avida ``.spop`` file.

    Returns
    -------
    tuple of (list[str], dict[str, list[str]])
        ``(header, avida_data)`` where *header* is the ordered list of
        field names and *avida_data* maps each field name to its list of
        raw string values (one entry per data row).  Missing trailing
        fields are filled with ``"NONE"``.

    Raises
    ------
    ValueError
        If the ``#format`` header line is missing from the spop text.
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
    create_ancestor_list: bool = True,
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
    create_ancestor_list : bool, default True
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

    # Add remaining Avida fields under their original names.
    for field in header:
        if field not in ("id", "parents", "update_born"):
            result_data[field] = avida_data[field]

    return pd.DataFrame(result_data)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Convert Avida .spop population snapshot data to Alife standard format.

Note that this CLI entrypoint is experimental and may be subject to change.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Avida .spop file to convert to Alife standard dataframe format.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to write Alife standard dataframe output to.",
    )
    parser.add_argument(
        "--output-engine",
        type=str,
        choices=["pandas", "polars"],
        default="pandas",
        help="DataFrame engine to use for writing the output file. Defaults to 'pandas'.",
    )
    parser.add_argument(
        "--no-ancestor-list",
        action="store_true",
        default=False,
        help="Exclude the ancestor_list column from the output.",
    )
    parser.add_argument(
        "--output-kwarg",
        action="append",
        dest="output_kwargs",
        type=str,
        default=[],
        help=(
            "Additional keyword arguments to pass to output engine call. "
            "Provide as 'key=value'. "
            "Specify multiple kwargs by using this flag multiple times. "
            "Arguments will be evaluated as Python expressions. "
            "Example: 'index=False'"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    import polars as pl

    configure_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()

    logging.info(f"reading Avida .spop data from {args.input_file}...")
    spop_str = pathlib.Path(args.input_file).read_text()

    with log_context_duration(
        "phyloframe.legacy.alifestd_from_avida_spop", logging.info
    ):
        logging.info("converting from Avida .spop format...")
        phylogeny_df = alifestd_from_avida_spop(
            spop_str,
            create_ancestor_list=not args.no_ancestor_list,
        )

    output_ext = os.path.splitext(args.output_file)[1]
    output_kwargs = eval_kwargs(args.output_kwargs)

    logging.info(
        f"writing alife-standard {output_ext} phylogeny data to "
        f"{args.output_file}...",
    )
    if args.output_engine == "polars":
        phylogeny_df = pl.from_pandas(phylogeny_df)
        dispatch_writer = {
            ".csv": pl.DataFrame.write_csv,
            ".fea": pl.DataFrame.write_ipc,
            ".feather": pl.DataFrame.write_ipc,
            ".pqt": pl.DataFrame.write_parquet,
            ".parquet": pl.DataFrame.write_parquet,
        }
    elif args.output_engine == "pandas":
        if output_ext == ".csv":
            output_kwargs.setdefault("index", False)
        dispatch_writer = {
            ".csv": pd.DataFrame.to_csv,
            ".fea": pd.DataFrame.to_feather,
            ".feather": pd.DataFrame.to_feather,
            ".pqt": pd.DataFrame.to_parquet,
            ".parquet": pd.DataFrame.to_parquet,
        }
    else:
        raise ValueError(f"unsupported output engine: {args.output_engine!r}")

    dispatch_writer[output_ext](
        phylogeny_df,
        args.output_file,
        **output_kwargs,
    )

    logging.info("done!")
