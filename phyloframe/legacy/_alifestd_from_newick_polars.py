import argparse
import logging
import os
import pathlib

import numpy as np
import polars as pl
import pyarrow as pa

from .._auxlib._configure_prod_logging import configure_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_from_newick import (
    _jit_build_label_buffer,
    _jit_parse_branch_lengths,
    _parse_newick_jit,
)


# Performance (as of 2026-03-15, 50k-node caterpillar tree, JIT-warmed):
#   with branch lengths: phyloframe ~0.007s vs treeswift ~0.34s (~0.02x)
#   without branch lengths: phyloframe ~0.003s vs treeswift ~0.21s (~0.01x)
# Uses JIT-compiled label byte-copy + float parsing + pyarrow zero-copy.
def alifestd_from_newick_polars(
    newick: str,
    *,
    branch_length_dtype: type = float,
    create_ancestor_list: bool = False,
) -> pl.DataFrame:
    """Convert a Newick format string to a phylogeny dataframe.

    Parses a Newick tree string and returns a polars DataFrame in alife
    standard format with columns: id, ancestor_id, taxon_label,
    origin_time_delta, and branch_length. Optionally includes
    ancestor_list.

    Parameters
    ----------
    newick : str
        A phylogeny in Newick format.
    branch_length_dtype : type, default float
        Dtype for branch length values. Use ``int`` to get nullable integer
        columns (``pl.Int64``). Missing branch lengths will be ``null`` for
        integer dtypes or ``NaN`` for float dtypes.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe in alife standard format.

    See Also
    --------
    alifestd_from_newick :
        Pandas-based implementation.
    alifestd_as_newick_asexual :
        Inverse conversion, from alife standard to Newick format.
    """
    newick = newick.strip()
    if not newick:
        columns = {
            "id": pl.Series([], dtype=pl.Int64),
            "ancestor_id": pl.Series([], dtype=pl.Int64),
            "taxon_label": pl.Series([], dtype=pl.Utf8),
            "origin_time_delta": pl.Series([], dtype=pl.Float64),
            "branch_length": pl.Series([], dtype=pl.Float64),
        }
        if create_ancestor_list:
            columns["ancestor_list"] = pl.Series([], dtype=pl.Utf8)
        return pl.DataFrame(columns)

    chars = np.frombuffer(newick.encode("ascii"), dtype=np.uint8)
    n = len(chars)

    (
        ids,
        ancestor_ids,
        label_starts,
        label_stops,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    ) = _parse_newick_jit(chars, n)

    # trim to actual sizes
    ids = ids[:num_nodes]
    ancestor_ids = ancestor_ids[:num_nodes]

    # build labels via JIT byte-copy + pyarrow zero-copy string array
    label_data, label_offsets = _jit_build_label_buffer(
        chars,
        label_starts[:num_nodes],
        label_stops[:num_nodes],
        num_nodes,
    )
    arrow_labels = pa.LargeStringArray.from_buffers(
        length=num_nodes,
        value_offsets=pa.py_buffer(label_offsets),
        data=pa.py_buffer(label_data),
    )
    labels_series = pl.Series("taxon_label", arrow_labels)

    # parse branch lengths directly in JIT (avoids Python string extraction)
    branch_lengths = _jit_parse_branch_lengths(
        chars,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    )

    # convert branch lengths to requested dtype with proper null handling
    np_dtype = np.dtype(branch_length_dtype)
    if np.issubdtype(np_dtype, np.integer):
        # NaN -> null, then cast to Int64
        bl_series = pl.Series(branch_lengths).fill_nan(None).cast(pl.Int64)
        otd_series = bl_series.clone()
    else:
        bl_series = pl.Series(branch_lengths)
        otd_series = pl.Series(branch_lengths.copy())

    columns = {
        "id": pl.Series(ids, dtype=pl.Int64),
        "ancestor_id": pl.Series(ancestor_ids, dtype=pl.Int64),
        "taxon_label": labels_series,
        "origin_time_delta": otd_series,
        "branch_length": bl_series,
    }

    if create_ancestor_list:
        from ._alifestd_make_ancestor_list_col_polars import (
            alifestd_make_ancestor_list_col_polars,
        )

        columns["ancestor_list"] = alifestd_make_ancestor_list_col_polars(
            pl.Series(ids, dtype=pl.Int64),
            pl.Series(ancestor_ids, dtype=pl.Int64),
        )

    return pl.DataFrame(columns)


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Convert Newick format phylogeny data to Alife standard format (Polars).

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
        help="Newick file to convert to Alife standard dataframe format.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to write Alife standard dataframe output to.",
    )
    parser.add_argument(
        "--branch-length-dtype",
        type=str,
        choices=["float", "int"],
        default="float",
        help="Dtype for branch length values. Defaults to 'float'.",
    )
    parser.add_argument(
        "--create-ancestor-list",
        action="store_true",
        default=False,
        help="Include an ancestor_list column in the output.",
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
            "Example: 'include_header=False'"
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


_dtype_lookup = {"float": float, "int": int}


if __name__ == "__main__":
    configure_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()

    logging.info(f"reading Newick data from {args.input_file}...")
    newick_str = pathlib.Path(args.input_file).read_text()

    with log_context_duration(
        "phyloframe.legacy.alifestd_from_newick_polars", logging.info
    ):
        logging.info("converting from Newick format...")
        phylogeny_df = alifestd_from_newick_polars(
            newick_str,
            branch_length_dtype=_dtype_lookup[args.branch_length_dtype],
            create_ancestor_list=args.create_ancestor_list,
        )

    output_ext = os.path.splitext(args.output_file)[1]
    output_kwargs = eval_kwargs(args.output_kwargs)
    dispatch_writer = {
        ".csv": pl.DataFrame.write_csv,
        ".fea": pl.DataFrame.write_ipc,
        ".feather": pl.DataFrame.write_ipc,
        ".pqt": pl.DataFrame.write_parquet,
        ".parquet": pl.DataFrame.write_parquet,
    }

    logging.info(
        f"writing alife-standard {output_ext} phylogeny data to "
        f"{args.output_file}...",
    )
    dispatch_writer[output_ext](
        phylogeny_df,
        args.output_file,
        **output_kwargs,
    )

    logging.info("done!")
