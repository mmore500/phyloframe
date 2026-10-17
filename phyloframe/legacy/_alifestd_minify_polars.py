import argparse
import functools
import logging
import os
import warnings

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._add_bool_arg import add_bool_arg
from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from .._auxlib._min_scalar_type_polars import min_scalar_type_polars
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_minify_asexual import (
    _alifestd_calc_minified_ancestor_offsets_asexual,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)
from ._alifestd_unfurl_traversal_preorder_polars import (
    alifestd_unfurl_traversal_preorder_polars,
)


def alifestd_minify_polars(
    phylogeny_df: pl.DataFrame,
    *,
    shrink_dtype: bool = True,
) -> pl.DataFrame:
    """Re-encode phylogeny_df compactly, replacing explicit ids with
    relative ancestor row offsets.

    The returned phylogeny dataframe will
    * be sorted in DFS preorder (i.e., organisms appear after their
    ancestor, with subtrees in contiguous rows, so that offspring tend to
    be close to their parents),
    * have no `id`, `ancestor_id`, or `ancestor_list` columns,
    * contain an `ancestor_offset` column, giving the number of rows back to
    each organism's ancestor (zero for roots).

    Each organism's ancestor is at row position `pos - ancestor_offset`.
    Use `alifestd_unminify_polars` to restore alife standard format.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows. Any existing `ancestor_offset` column
        will be overwritten.
    shrink_dtype : bool, default True
        If True, `ancestor_offset` will use the smallest signed integer dtype
        that can hold all offsets. Otherwise, it will have the same dtype as
        `ancestor_id`.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The minified phylogeny.

    Raises
    ------
    NotImplementedError
        If ids are not contiguous or rows are not topologically sorted.

    See Also
    --------
    alifestd_minify_asexual :
        Pandas-based implementation.
    alifestd_unminify_polars :
        Inverse operation, restoring alife standard format.
    alifestd_to_working_format_polars :
        Prepare contiguous ids and topologically sorted rows.
    """
    if "ancestor_offset" in phylogeny_df.lazy().collect_schema().names():
        warnings.warn(
            "alifestd_minify_polars overwrites existing ancestor_offset",
        )

    logging.info(
        "- alifestd_minify_polars: adding ancestor_id col...",
    )
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    logging.info(
        "- alifestd_minify_polars: checking contiguous ids...",
    )
    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError(
            "non-contiguous ids not supported",
        )

    logging.info(
        "- alifestd_minify_polars: checking topological sort...",
    )
    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_minify_polars: calculating preorder traversal...",
    )
    order = alifestd_unfurl_traversal_preorder_polars(phylogeny_df)

    logging.info(
        "- alifestd_minify_polars: calculating ancestor offsets...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )
    ancestor_offsets = pl.Series(
        _alifestd_calc_minified_ancestor_offsets_asexual(
            ancestor_ids,
            order.astype(ancestor_ids.dtype),
        ),
    )
    if shrink_dtype:
        offset_dtype = min_scalar_type_polars(-(ancestor_offsets.max() or 1))
        logging.info(
            "- alifestd_minify_polars: "
            f"shrinking ancestor_offset dtype to {offset_dtype}...",
        )
        ancestor_offsets = ancestor_offsets.cast(offset_dtype)

    logging.info(
        "- alifestd_minify_polars: reordering rows...",
    )
    return (
        phylogeny_df.select(pl.all().gather(order))
        .with_columns(ancestor_offset=ancestor_offsets)
        .select(pl.exclude("id", "ancestor_id", "ancestor_list"))
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Re-encode phylogeny compactly, replacing explicit ids with relative ancestor row offsets.

The output phylogeny will
- be sorted in DFS preorder (i.e., organisms appear after their ancestor, with subtrees in contiguous rows, so that offspring tend to be close to their parents),
- have no `id`, `ancestor_id`, or `ancestor_list` columns,
- contain an `ancestor_offset` column, giving the number of rows back to each organism's ancestor (zero for roots).

By default, `ancestor_offset` uses the smallest signed integer dtype that can hold all offsets.
Use `--no-shrink-dtype` to instead keep the `ancestor_id` dtype.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires contiguous ids and topologically sorted rows. Otherwise, use `alifestd_to_working_format_polars` first or the pandas op CLI.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_minify_asexual :
    CLI entrypoint for Pandas-based implementation.
phyloframe.legacy._alifestd_unminify_polars :
    CLI entrypoint for inverse operation, restoring alife standard format.
"""


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False,
        allow_abbrev=False,
        description=format_cli_description(_raw_description),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser = _add_parser_base(
        parser=parser,
        dfcli_module="phyloframe.legacy._alifestd_minify_polars",
        dfcli_version=get_phyloframe_version(),
    )
    add_bool_arg(
        parser,
        "shrink-dtype",
        default=True,
        help=(
            "use the smallest integer dtype that can hold `ancestor_offset` "
            "values (default: True)"
        ),
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_minify_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_minify_polars,
                    shrink_dtype=args.shrink_dtype,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
