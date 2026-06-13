import argparse
import ast
import logging
import os
import pathlib
import types
import typing
import warnings

import numpy as np
import polars as pl
import pyarrow as pa

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._configure_prod_logging import configure_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._find_equivalent_numpy_dtype_polars import (
    find_equivalent_numpy_dtype_polars,
)
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._min_scalar_type_polars import min_scalar_type_polars
from ._alifestd_from_newick import (
    _jit_build_label_buffer,
    _jit_parse_branch_lengths,
    _parse_newick_jit,
)


# Performance (as of 2026-03-15, 200k-node caterpillar tree, JIT-warmed):
#   with branch lengths: phyloframe ~0.06s vs treeswift ~1.4s (~0.04x)
#   without branch lengths: phyloframe ~0.02s vs treeswift ~0.8s (~0.03x)
# Uses JIT-compiled label byte-copy + float parsing + pyarrow zero-copy.
def alifestd_from_newick_polars(
    newick: str,
    *,
    allow_forest: typing.Optional[bool] = None,
    branch_length_dtype: type = float,
    create_ancestor_list: bool = False,
    dtype_id: typing.Optional[pl.datatypes.DataType] = pl.Int64,
    replace_unquoted: typing.Mapping[str, str] = types.MappingProxyType({}),
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
    allow_forest : bool or None, default None
        Policy for a Newick string holding multiple ``;``-terminated trees
        (a forest). ``None`` parses the forest but warns; ``True`` parses it
        silently; ``False`` raises ``ValueError`` unless there is a single
        tree.
    branch_length_dtype : type, default float
        Dtype for branch length values. Use ``int`` to get nullable integer
        columns (``pl.Int64``). Missing branch lengths will be ``null`` for
        integer dtypes or ``NaN`` for float dtypes.
    create_ancestor_list : bool, default False
        If True, include an ``ancestor_list`` column in the result.
    dtype_id : pl.DataType or None, default pl.Int64
        Polars dtype for the ``id`` and ``ancestor_id`` columns. If None, the
        smallest signed integer dtype that can hold all node ids is chosen
        automatically based on the node count of the Newick string.
    replace_unquoted : Mapping[str, str], optional
        Character substitutions to apply to *unquoted* taxon labels only,
        leaving quoted labels verbatim. Keys must be single characters.
        Pass ``{"_": " "}`` to follow the strict Newick convention in which
        an unquoted underscore denotes a space.

    Returns
    -------
    pl.DataFrame
        Phylogeny dataframe in alife standard format.

    Notes
    -----
    By default, unquoted underscores in taxon labels are preserved literally;
    they are *not* converted to spaces. This diverges from the strict Newick
    convention (in which an unquoted ``_`` denotes a space), but matches the
    round-trip behavior of ``alifestd_as_newick_asexual``. Pass
    ``replace_unquoted={"_": " "}`` to follow the strict convention.

    See Also
    --------
    alifestd_from_newick :
        Pandas-based implementation.
    alifestd_as_newick_asexual :
        Inverse conversion, from alife standard to Newick format.
    """
    newick = newick.strip()
    if dtype_id is None:
        # the parser assigns one node id per '(' and per ',', plus one root
        # per tree (';'-terminated); size the dtype from that node-count
        # upper bound rather than commas alone to avoid overflow
        node_count = newick.count("(") + newick.count(",") + newick.count(";")
        pl_dtype_id = min_scalar_type_polars(-max(node_count, 1))
    else:
        pl_dtype_id = dtype_id

    if not newick:
        columns = {
            "id": pl.Series([], dtype=pl_dtype_id),
            "ancestor_id": pl.Series([], dtype=pl_dtype_id),
            "taxon_label": pl.Series([], dtype=pl.Utf8),
            "origin_time_delta": pl.Series([], dtype=pl.Float64),
            "branch_length": pl.Series([], dtype=pl.Float64),
        }
        if create_ancestor_list:
            columns["ancestor_list"] = pl.Series([], dtype=pl.Utf8)
        return pl.DataFrame(columns)

    chars = np.frombuffer(newick.encode("ascii"), dtype=np.uint8)
    n = len(chars)

    np_dtype_id = find_equivalent_numpy_dtype_polars(pl_dtype_id)

    (
        ids,
        ancestor_ids,
        label_starts,
        label_stops,
        label_quoted,
        bl_starts,
        bl_stops,
        bl_node_ids,
        num_nodes,
        num_bls,
    ) = _parse_newick_jit(chars, n, np_dtype_id)

    if not allow_forest and np.count_nonzero(ancestor_ids == ids) > 1:
        if allow_forest is False:
            raise ValueError(
                "Newick string contains a forest of multiple trees; "
                "pass allow_forest=True to allow.",
            )
        warnings.warn(
            "Newick string contains a forest of multiple trees; pass "
            "allow_forest=True to silence this warning or allow_forest=False "
            "to require a single tree.",
        )

    # build labels via JIT byte-copy + pyarrow zero-copy string array
    label_data, label_offsets = _jit_build_label_buffer(
        chars,
        label_starts,
        label_stops,
        num_nodes,
    )
    arrow_labels = pa.LargeStringArray.from_buffers(
        length=num_nodes,
        value_offsets=pa.py_buffer(label_offsets),
        data=pa.py_buffer(label_data),
    )
    labels_series = pl.Series("taxon_label", arrow_labels)

    # fix up labels in a single lazy pass: collapse escaped '' inside quoted
    # labels, and apply replace_unquoted substitutions to unquoted labels.
    # only run when needed -- the common parse (no quoted labels, no
    # substitutions) skips this entirely. (the str ops scan every label, so
    # the guard is a real saving, not redundant with polars' optimizer.)
    if replace_unquoted and any(len(key) != 1 for key in replace_unquoted):
        raise ValueError("replace_unquoted keys must be single characters")
    quoted_any = bool(label_quoted.any())
    if quoted_any or replace_unquoted:
        labels_series = (
            pl.LazyFrame(
                {
                    "taxon_label": labels_series,
                    "__quoted": pl.Series(label_quoted.astype(bool)),
                },
            )
            .select(
                pl.when(pl.col("__quoted"))
                .then(
                    pl.col("taxon_label").str.replace_all(
                        "''", "'", literal=True
                    )
                )
                .otherwise(
                    pl.col("taxon_label").str.replace_many(
                        dict(replace_unquoted)
                    )
                )
                .alias("taxon_label"),
            )
            .collect()
            .to_series()
        )

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
        "id": pl.Series(ids, dtype=pl_dtype_id),
        "ancestor_id": pl.Series(ancestor_ids, dtype=pl_dtype_id),
        "taxon_label": labels_series,
        "origin_time_delta": otd_series,
        "branch_length": bl_series,
    }

    if create_ancestor_list:
        from ._alifestd_make_ancestor_list_col_polars import (
            alifestd_make_ancestor_list_col_polars,
        )

        columns["ancestor_list"] = alifestd_make_ancestor_list_col_polars(
            pl.Series(ids, dtype=pl_dtype_id),
            pl.Series(ancestor_ids, dtype=pl_dtype_id),
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
    add_bool_arg(
        parser,
        "create-ancestor-list",
        default=False,
        help="Include an ancestor_list column in the output.",
    )
    add_bool_arg(
        parser,
        "allow-forest",
        default=None,
        help=(
            "Allow a multi-tree (forest) Newick string. Unset warns; "
            "--allow-forest is silent; --no-allow-forest requires a single "
            "tree."
        ),
    )
    parser.add_argument(
        "--replace-unquoted",
        type=str,
        default="{}",
        metavar="MAPPING",
        help=(
            "Mapping of single-character substitutions to apply to unquoted "
            "taxon labels only (quoted labels are left verbatim), given as a "
            "Python dict literal. "
            "Example: \"{'_': ' '}\" maps unquoted underscores to spaces."
        ),
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
            allow_forest=args.allow_forest,
            branch_length_dtype=_dtype_lookup[args.branch_length_dtype],
            create_ancestor_list=args.create_ancestor_list,
            replace_unquoted=ast.literal_eval(args.replace_unquoted),
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
