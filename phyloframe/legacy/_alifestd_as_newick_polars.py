import argparse
import logging
import os
import pathlib
import typing

import numpy as np
import polars as pl
from tqdm import tqdm

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._eval_kwargs import eval_kwargs
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_postorder_asexual import (
    _alifestd_unfurl_traversal_postorder_asexual_fast_path,
)


def _format_newick_reprs_polars(
    labels: np.ndarray,
    origin_time_deltas: np.ndarray,
) -> np.ndarray:
    """Vectorize newick repr formatting using Polars expressions.

    Equivalent to calling _format_newick_repr per-element but batched,
    using the Polars expression engine for string processing.
    """
    df = pl.DataFrame(
        {
            "label": pl.Series(labels, dtype=pl.Utf8),
            "otd": origin_time_deltas,
        },
    )
    result = (
        df.lazy()
        .with_columns(
            # Quote labels containing unsafe Newick symbols
            label=pl.when(
                pl.col("label").str.contains(r"[;()\[\],:']"),
            )
            .then(pl.lit("'") + pl.col("label") + pl.lit("'"))
            .otherwise(pl.col("label")),
            # Strip trailing zeros from decimal origin_time_deltas
            otd_str=pl.when(pl.col("otd").is_nan())
            .then(pl.lit(None))
            .otherwise(
                pl.col("otd")
                .cast(pl.Utf8)
                .str.replace(r"(\.\d*?)0+$", "$1")
                .str.replace(r"\.$", ""),
            ),
        )
        .with_columns(
            repr=pl.when(pl.col("otd_str").is_null())
            .then(pl.col("label"))
            .otherwise(pl.col("label") + ":" + pl.col("otd_str")),
        )
        .select("repr")
        .collect()
        .to_series()
        .to_numpy()
    )
    return result


def _build_newick_postorder(
    ids: np.ndarray,
    node_reprs: np.ndarray,
    ancestor_ids: np.ndarray,
    n_nodes: int,
    *,
    progress_wrap: typing.Callable,
) -> str:
    """Build Newick string from postorder-traversed arrays.

    Uses the postorder traversal to build tree structure (parent-child
    relationships), then serializes via iterative DFS with a flat token
    list joined once at the end. This avoids the O(n^2) string
    concatenation that occurs with deep trees (e.g., caterpillar
    topologies) in the naive approach.
    """
    # Pass 1: postorder traversal to build tree structure.
    # Use list indexing (contiguous IDs) instead of dict for speed.
    children: typing.List[typing.Optional[typing.List[int]]] = [None] * n_nodes
    roots = []
    for id_, ancestor_id in progress_wrap(zip(ids, ancestor_ids)):
        if id_ == ancestor_id:
            roots.append(id_)
        else:
            kids = children[ancestor_id]
            if kids is None:
                children[ancestor_id] = [id_]
            else:
                kids.append(id_)

    logging.info(f"finalizing {len(roots)} subtrees...")

    # Pass 2: iterative DFS to emit tokens in Newick order.
    parts = []
    parts_append = parts.append
    for root_idx, root in enumerate(roots):
        if root_idx:
            parts_append(";\n")
        # Stack: (node_id, child_position)
        #   -1 = not yet started children
        #   k  = next child index to push
        stack = [(root, -1)]
        while stack:
            node, child_pos = stack[-1]
            kids = children[node]
            if kids is None:
                parts_append(node_reprs[node])
                stack.pop()
            elif child_pos == -1:
                parts_append("(")
                stack[-1] = (node, 1)
                stack.append((kids[0], -1))
            elif child_pos < len(kids):
                parts_append(",")
                stack[-1] = (node, child_pos + 1)
                stack.append((kids[child_pos], -1))
            else:
                parts_append(")")
                parts_append(node_reprs[node])
                stack.pop()

    parts_append(";")
    return "".join(parts)


def alifestd_as_newick_polars(
    phylogeny_df: pl.DataFrame,
    *,
    taxon_label: typing.Optional[str] = None,
    progress_wrap: typing.Callable = lambda x: x,
) -> str:
    """Convert phylogeny dataframe to Newick format.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        Phylogeny dataframe in Alife standard format.
    taxon_label : str, optional
        Column to use for taxon labels, by default None.
    progress_wrap : typing.Callable, optional
        Pass tqdm or equivalent to display a progress bar.

    See Also
    --------
    alifestd_as_newick_asexual :
        Pandas-based implementation.
    """
    logging.info(
        "creating newick string for alifestd polars df",
    )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return ";"

    logging.info("adding ancestor id column, if not present")
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema_names = phylogeny_df.lazy().collect_schema().names()

    if "ancestor_id" not in schema_names:
        raise ValueError("only asexual phylogenies supported")

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not yet supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "polars topological sort not yet implemented",
        )

    ancestor_ids = (
        phylogeny_df.lazy()
        .select(pl.col("ancestor_id").cast(pl.Int64))
        .collect()
        .to_series()
        .to_numpy()
    )
    n = len(ancestor_ids)
    ids = np.arange(n)

    logging.info("setting up `origin_time_delta`...")
    if "origin_time_delta" in schema_names:
        logging.info("... already present!")
        origin_time_deltas = (
            phylogeny_df.lazy()
            .select("origin_time_delta")
            .collect()
            .to_series()
            .to_numpy()
        )
    elif "origin_time" in schema_names:
        logging.info("... calculating from `origin_time`...")
        origin_times = (
            phylogeny_df.lazy()
            .select("origin_time")
            .collect()
            .to_series()
            .to_numpy()
        )
        origin_time_deltas = (
            origin_times - origin_times[ancestor_ids.astype(int)]
        )
    else:
        logging.info("... marking null")
        origin_time_deltas = np.full(n, np.nan)

    logging.info("calculating postorder traversal order...")
    postorder_index = _alifestd_unfurl_traversal_postorder_asexual_fast_path(
        ancestor_ids,
    )

    logging.info("preparing labels...")
    if taxon_label is not None:
        labels = (
            phylogeny_df.lazy()
            .select(pl.col(taxon_label).cast(pl.Utf8))
            .collect()
            .to_series()
            .to_numpy()
        )
    else:
        labels = np.full(n, "", dtype=object)

    logging.info("formatting node representations...")
    node_reprs = _format_newick_reprs_polars(labels, origin_time_deltas)

    logging.info("creating newick string...")
    result = _build_newick_postorder(
        ids[postorder_index],
        node_reprs,
        ancestor_ids[postorder_index],
        n,
        progress_wrap=progress_wrap,
    )

    logging.info(f"{len(result)=} {result[:20]=}")
    return result


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()})

Convert Alife standard phylogeny data to Newick format, using polars.

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
        ".csv": pl.scan_csv,
        ".fea": pl.scan_ipc,
        ".feather": pl.scan_ipc,
        ".pqt": pl.scan_parquet,
        ".parquet": pl.scan_parquet,
    }

    logging.info(
        f"reading alife-standard {input_ext} phylogeny data from "
        f"{args.input_file}...",
    )
    phylogeny_df = dispatch_reader[input_ext](
        args.input_file,
        **eval_kwargs(args.input_kwargs),
    )

    with log_context_duration(
        "phyloframe.legacy.alifestd_as_newick_polars", logging.info
    ):
        logging.info("converting to Newick format...")
        newick_str = alifestd_as_newick_polars(
            phylogeny_df, progress_wrap=tqdm, taxon_label=args.taxon_label
        )

    logging.info(f"writing Newick-formatted data to {args.output_file}...")
    pathlib.Path(args.output_file).write_text(newick_str)

    logging.info("done!")
