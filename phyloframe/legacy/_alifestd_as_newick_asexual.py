import argparse
import logging
import os
import typing

import more_itertools as mit
import numpy as np
import opytional as opyt
import pandas as pd
import polars as pl
from tqdm import tqdm

from .._auxlib._add_compression_cli_arg import add_compression_cli_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._write_text_with_compression import (
    write_text_with_compression,
)
from ._alifestd_has_contiguous_ids import alifestd_has_contiguous_ids
from ._alifestd_mark_origin_time_delta_asexual import (
    alifestd_mark_origin_time_delta_asexual,
)
from ._alifestd_try_add_ancestor_id_col import alifestd_try_add_ancestor_id_col
from ._alifestd_unfurl_traversal_postorder_asexual import (
    alifestd_unfurl_traversal_postorder_asexual,
)

# adapted from https://stackoverflow.com/a/3939381/17332200
# whitespace is included so labels with spaces are quoted: an unquoted space
# is non-standard Newick and, under the underscore convention, ambiguous
_UNSAFE_SYMBOLS = ";(),[]:' \t\n"


def _format_newick_repr(
    taxon_label: str,
    origin_time_delta: str,
    unsafe_table: dict,
) -> str:
    # adapted from https://github.com/niemasd/TreeSwift/blob/63b8979fb5e616ba89079d44e594682683c1365e/treeswift/Node.py#L129
    label = taxon_label

    if label.translate(unsafe_table) != label:
        # quote the label, doubling any embedded single quotes per the
        # Newick convention so the label round-trips through the parser
        label = label.replace("'", "''").join("''")

    if origin_time_delta:  # empty string denotes a missing branch length
        if "." in origin_time_delta:
            origin_time_delta = origin_time_delta.rstrip("0").rstrip(".")
        label = f"{label}:{origin_time_delta}"

    return label


def _build_newick_string(
    ids: np.ndarray,
    labels: np.ndarray,
    origin_time_deltas: np.ndarray,
    ancestor_ids: np.ndarray,
    *,
    unsafe_symbols: str,
    progress_wrap: typing.Callable,
) -> str:
    unsafe_table = str.maketrans("", "", unsafe_symbols)
    # use empty string (never a valid branch length) to mark a missing
    # origin_time_delta, avoiding a "nan"/"<NA>" sentinel that could collide
    # with a taxon literally named "nan"
    origin_time_delta_strs = np.where(
        pd.isna(origin_time_deltas), "", origin_time_deltas.astype(str)
    )

    child_newick_reprs = dict()
    for id_, taxon_label, otd_str, ancestor_id in progress_wrap(
        zip(ids, labels, origin_time_delta_strs, ancestor_ids)
    ):
        newick_repr = _format_newick_repr(taxon_label, otd_str, unsafe_table)

        children_reprs = child_newick_reprs.pop(id_, None)
        if children_reprs is not None:
            newick_repr = f"({','.join(children_reprs)}){newick_repr}"

        child_newick_reprs.setdefault(ancestor_id, []).append(newick_repr)

    logging.info(f"finalizing {len(child_newick_reprs)} subtrees...")
    return ";\n".join(map(mit.one, child_newick_reprs.values())) + ";"


# Performance (as of 2026-03-01, 200k-node caterpillar tree):
#   phyloframe ~9s vs treeswift ~5s
def alifestd_as_newick_asexual(
    phylogeny_df: pd.DataFrame,
    mutate: bool = False,
    *,
    taxon_label: typing.Optional[str] = None,
    unsafe_symbols: str = _UNSAFE_SYMBOLS,
    progress_wrap: typing.Callable = lambda x: x,
) -> str:
    """Convert phylogeny dataframe to Newick format.

    Parameters
    ----------
    phylogeny_df : pd.DataFrame
        Phylogeny dataframe in Alife standard format.
    mutate : bool, optional
        Allow in-place mutations of the input dataframe, by default False.
    taxon_label : str, optional
        Column to use for taxon labels, by default None.
    unsafe_symbols : str, optional
        Characters that force a taxon label to be single-quoted when present.
        Defaults to the Newick-reserved symbols (and whitespace).
    progress_wrap : typing.Callable, optional
        Pass tqdm or equivalent to display a progress bar.
    """

    logging.info(
        "creating newick string for alifestd df "
        f"with shape {phylogeny_df.shape}",
    )

    if not mutate:
        phylogeny_df = phylogeny_df.copy()

    logging.info("adding ancestor id column, if not present")
    phylogeny_df = alifestd_try_add_ancestor_id_col(phylogeny_df, mutate=True)

    if alifestd_has_contiguous_ids(phylogeny_df):
        phylogeny_df.reset_index(drop=True, inplace=True)
    else:
        phylogeny_df.index = phylogeny_df["id"]

    logging.info("setting up `origin_time_delta` column...")
    if "origin_time_delta" in phylogeny_df.columns:
        logging.info("... already present!")
    elif "origin_time" in phylogeny_df.columns:
        logging.info("... calculating from `origin_time`...")
        phylogeny_df = alifestd_mark_origin_time_delta_asexual(
            phylogeny_df, mutate=True
        )
    else:
        logging.info("... marking null")
        phylogeny_df["origin_time_delta"] = np.nan

    logging.info("calculating postorder traversal order...")
    postorder_ids = alifestd_unfurl_traversal_postorder_asexual(phylogeny_df)

    logging.info("preparing labels...")
    phylogeny_df["__phyloframe_label"] = opyt.apply_if_or_value(
        taxon_label, phylogeny_df.__getitem__, ""
    )
    phylogeny_df["__phyloframe_label"] = phylogeny_df[
        "__phyloframe_label"
    ].astype(str)

    logging.info("reshaping data...")
    reshaped = (
        phylogeny_df.loc[
            postorder_ids,
            ["id", "__phyloframe_label", "origin_time_delta", "ancestor_id"],
        ]
        .to_numpy()
        .T
    )

    logging.info("creating newick string...")
    result = _build_newick_string(
        *reshaped,
        unsafe_symbols=unsafe_symbols,
        progress_wrap=progress_wrap,
    )

    logging.info(f"{len(result)=} {result[:20]=}")
    return result


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Convert Alife standard phylogeny data to Newick format.

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
        help="Alife standard dataframe file to convert to Newick format.",
    )
    parser.add_argument(
        "--input-engine",
        type=str,
        choices=["pandas", "polars"],
        default="pandas",
        help="DataFrame engine to use for reading the input file. Defaults to 'pandas'.",
    )
    parser.add_argument(
        "--input-kwarg",
        action="append",
        dest="input_kwargs",
        type=str,
        default=[],
        help=(
            "Additional keyword arguments to pass to input engine call. "
            "Provide as 'key=value'. "
            "Specify multiple kwargs by using this flag multiple times. "
            "Arguments will be evaluated as Python expressions. "
            "Example: 'infer_schema_length=None'"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Path to write Newick-formatted output to.",
    )
    parser.add_argument(
        "-l",
        "--taxon-label",
        type=str,
        help="Name of column to use as taxon label.",
        required=False,
    )
    parser.add_argument(
        "--unsafe-symbols",
        type=str,
        default=_UNSAFE_SYMBOLS,
        help=(
            "Characters that force a taxon label to be single-quoted when "
            "present. Defaults to the Newick-reserved symbols and whitespace."
        ),
    )
    add_compression_cli_arg(parser)
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=get_phyloframe_version(),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args = parser.parse_args()
    input_ext = os.path.splitext(args.input_file)[1]
    dispatch_reader = {
        "pandas+.csv": pd.read_csv,
        "pandas+.fea": pd.read_feather,
        "pandas+.feather": pd.read_feather,
        "pandas+.pqt": pd.read_parquet,
        "pandas+.parquet": pd.read_parquet,
        "polars+.csv": pl.read_csv,
        "polars+.fea": pl.read_ipc,
        "polars+.feather": pl.read_ipc,
        "polars+.pqt": pl.read_parquet,
        "polars+.parquet": pl.read_parquet,
    }

    logging.info(
        f"reading alife-standard {input_ext} phylogeny data from "
        f"{args.input_file}...",
    )
    phylogeny_df = dispatch_reader[f"{args.input_engine}+{input_ext}"](
        args.input_file,
        **eval_kwargs(args.input_kwargs),
    )

    if args.input_engine == "polars":
        with log_context_duration("pl.DataFrame.to_pandas", logging.info):
            phylogeny_df = phylogeny_df.to_pandas()

    with log_context_duration(
        "phyloframe.legacy.alifestd_as_newick_asexual", logging.info
    ):
        logging.info("converting to Newick format...")
        newick_str = alifestd_as_newick_asexual(
            phylogeny_df,
            progress_wrap=tqdm,
            taxon_label=args.taxon_label,
            unsafe_symbols=args.unsafe_symbols,
        )

    logging.info(f"writing Newick-formatted data to {args.output_file}...")
    write_text_with_compression(args.output_file, newick_str, args.compression)

    logging.info("done!")
