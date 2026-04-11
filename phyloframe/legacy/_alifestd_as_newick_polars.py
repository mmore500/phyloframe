import argparse
import logging
import os
import typing

import opytional as opyt
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
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_csr_children_asexual import (
    _alifestd_mark_csr_children_asexual_fast_path,
)
from ._alifestd_mark_csr_offsets_asexual import (
    _alifestd_mark_csr_offsets_asexual_fast_path,
)
from ._alifestd_mark_node_depth_asexual import (
    _alifestd_calc_node_depth_asexual_contiguous,
)
from ._alifestd_mark_num_children_asexual import (
    _alifestd_mark_num_children_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_postorder_contiguous_asexual import (
    _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit,
)


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
    logging.info("creating newick string for alifestd polars df")

    phylogeny_df = phylogeny_df.lazy()

    if phylogeny_df.limit(1).collect().is_empty():
        return ";"

    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    schema = phylogeny_df.collect_schema()
    schema_names = schema.names()

    if "ancestor_id" not in schema_names:
        raise ValueError("only asexual phylogenies supported")

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):

        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):

        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    ancestor_ids = (
        phylogeny_df.select(pl.col("ancestor_id"))
        .collect()
        .to_series()
        .to_numpy()
    )
    len(ancestor_ids)

    if "node_depth" not in schema_names:
        node_depth = _alifestd_calc_node_depth_asexual_contiguous(ancestor_ids)
        phylogeny_df = phylogeny_df.with_columns(
            node_depth=pl.Series(node_depth),
        )
    else:
        node_depth = (
            phylogeny_df.select(pl.col("node_depth"))
            .collect()
            .to_series()
            .to_numpy()
        )

    logging.info("setting up `origin_time_delta`...")
    if "origin_time_delta" in schema_names:
        logging.info("... already present!")
        origin_time_deltas = pl.col("origin_time_delta")
    elif "origin_time" in schema_names:
        logging.info("... calculating from `origin_time`...")
        phylogeny_df = phylogeny_df.with_columns(
            ancestor_origin_time=(
                pl.col("origin_time").gather(pl.col("ancestor_id"))
            ),
        )
        origin_time_deltas = pl.col("origin_time") - pl.col(
            "ancestor_origin_time"
        )
    else:
        logging.info("... marking null")
        origin_time_deltas = pl.lit(None).cast(pl.Int64)

    phylogeny_df = phylogeny_df.with_columns(
        origin_time_deltas=origin_time_deltas,
    )

    logging.info("gathering num_children...")
    if "num_children" not in schema_names:
        num_children = _alifestd_mark_num_children_asexual_fast_path(
            ancestor_ids,
        )
    else:
        num_children = (
            phylogeny_df.select("num_children")
            .collect()
            .to_series()
            .to_numpy()
        )

    logging.info("gathering csr_offsets...")
    if "csr_offsets" not in schema_names:
        csr_offsets = _alifestd_mark_csr_offsets_asexual_fast_path(
            ancestor_ids
        )
    else:
        csr_offsets = (
            phylogeny_df.select("csr_offsets").collect().to_series().to_numpy()
        )

    logging.info("gathering csr_children...")
    if "csr_children" not in schema_names:
        csr_children = _alifestd_mark_csr_children_asexual_fast_path(
            ancestor_ids,
            csr_offsets,
        )
    else:
        csr_children = (
            phylogeny_df.select("csr_children")
            .collect()
            .to_series()
            .to_numpy()
        )

    logging.info("calculating postorder traversal order...")
    postorder_index = (
        _alifestd_unfurl_traversal_postorder_contiguous_asexual_jit(
            ancestor_ids,
            csr_offsets,
            csr_children,
            num_children,
        )
    )

    logging.info("gathering postorder...")
    select_cols = ["id", "ancestor_id", "node_depth", "origin_time_deltas"]
    phylogeny_df = (
        phylogeny_df.select(
            *select_cols,
            *opyt.apply_if_or_value(
                taxon_label,
                lambda x: [x] if x not in select_cols else [],
                [],
            ),
        )
        .collect()[postorder_index]
        .lazy()
    )

    logging.info("preparing labels...")
    label_cname = "__as_newick_taxon_label"
    if taxon_label is not None:
        phylogeny_df = phylogeny_df.with_columns(
            **{label_cname: pl.col(taxon_label).cast(pl.String)},
        )
        dtype = schema.get(taxon_label)
        if dtype in (pl.String, pl.Utf8, pl.Object):
            phylogeny_df = phylogeny_df.with_columns(
                **{
                    label_cname: pl.when(
                        pl.col(label_cname).str.contains(r"[:();,]")
                    )
                    .then(pl.lit("'") + pl.col(label_cname) + pl.lit("'"))
                    .otherwise(pl.col(label_cname))
                }
            )
        taxon_label_expr = pl.col(label_cname)
    else:
        taxon_label_expr = pl.lit("")

    logging.info("marking node depth diffs...")
    phylogeny_df = phylogeny_df.with_columns(
        raw_diff=pl.col("node_depth").diff()
    ).with_columns(
        node_depth_diffs=pl.coalesce(pl.col("raw_diff"), pl.col("node_depth")),
        needs_comma=(pl.col("raw_diff") >= 0).fill_null(False),
    )

    logging.info("creating tokens...")
    abs_diff = pl.col("node_depth_diffs").abs().cast(pl.UInt64)
    parens_str = (
        pl.when(pl.col("node_depth_diffs") > 0)
        .then(pl.lit("").str.pad_end(abs_diff, "("))
        .when(pl.col("node_depth_diffs") < 0)
        .then(pl.lit(")"))
        .otherwise(pl.lit(""))
    )
    comma_str = (
        pl.when(pl.col("needs_comma")).then(pl.lit(",")).otherwise(pl.lit(""))
    )
    root_str = (
        pl.when(pl.col("ancestor_id") == pl.col("id"))
        .then(pl.lit(";\n"))
        .otherwise(pl.lit(""))
    )

    # Format branch lengths: strip trailing zeros after decimal point
    origin_time_deltas_dtype = phylogeny_df.collect_schema()[
        "origin_time_deltas"
    ]
    formatted_deltas = pl.col("origin_time_deltas").cast(pl.String)
    if origin_time_deltas_dtype.is_float():
        formatted_deltas = formatted_deltas.str.strip_chars_end(
            "0"
        ).str.strip_chars_end(".")

    branch_len_str = (
        pl.when(pl.col("origin_time_deltas").is_not_null())
        .then(pl.lit(":") + formatted_deltas)
        .otherwise(pl.lit(""))
    )

    phylogeny_df = phylogeny_df.with_columns(
        _newick_token=(
            comma_str
            + parens_str
            + taxon_label_expr
            + branch_len_str
            + root_str
        ),
    )

    logging.info("creating newick string...")
    result = (
        phylogeny_df.select(pl.col("_newick_token").str.join("").str.head(-1))
        .collect()
        .item()
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
    write_text_with_compression(args.output_file, newick_str, args.compression)

    logging.info("done!")
