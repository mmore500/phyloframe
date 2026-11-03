import argparse
import functools
import logging
import os

import joinem
from joinem._dataframe_cli import _add_parser_base, _run_dataframe_cli
import polars as pl

from .._auxlib._begin_prod_logging import begin_prod_logging
from .._auxlib._format_cli_description import format_cli_description
from .._auxlib._get_phyloframe_version import get_phyloframe_version
from .._auxlib._log_context_duration import log_context_duration
from ._alifestd_has_contiguous_ids_polars import (
    alifestd_has_contiguous_ids_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_num_mrca_pairs_asexual import (
    _alifestd_mark_num_mrca_pairs_asexual_fast_path,
)
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_num_mrca_pairs_polars(
    phylogeny_df: pl.DataFrame,
    *,
    mark_as: str = "num_mrca_pairs",
) -> pl.DataFrame:
    """Add column with number of leaf pairs whose most recent common
    ancestor is each node.

    For a node with `n` descendant leaves whose children have `c_i`
    descendant leaves each, this count is `C(n, 2) - sum_i C(c_i, 2)`.
    So, leaves and unifurcations have count zero, and counts sum to
    `C(num_leaves, 2)` over each tree. Polytomies are supported.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame or polars.LazyFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with contiguous ids and
        topologically sorted rows.
    mark_as : str, default "num_mrca_pairs"
        Output column name.

    Returns
    -------
    polars.DataFrame or polars.LazyFrame
        The phylogeny with an added mark column.

    Raises
    ------
    NotImplementedError
        If `phylogeny_df` is not asexual, has non-contiguous ids, or is not
        topologically sorted.

    See Also
    --------
    alifestd_mark_num_mrca_pairs_asexual :
        Pandas-based implementation.
    alifestd_mark_num_leaves_polars :
        Counts descendant leaves of each node.
    alifestd_find_pair_mrca_id_polars :
        Finds most recent common ancestor of a pair of taxa.
    """
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)
    if "ancestor_id" not in phylogeny_df.lazy().collect_schema().names():
        raise NotImplementedError(
            "alifestd_mark_num_mrca_pairs_polars only supports asexual "
            "phylogenies.",
        )

    if phylogeny_df.lazy().limit(1).collect().is_empty():
        return phylogeny_df.with_columns(
            pl.lit(0, dtype=pl.Int64).alias(mark_as)
        )

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError(
            "non-topologically-sorted data not supported",
        )

    logging.info(
        "- alifestd_mark_num_mrca_pairs_polars: extracting ancestor ids...",
    )
    ancestor_ids = (
        phylogeny_df.lazy()
        .select("ancestor_id")
        .collect()
        .to_series()
        .to_numpy()
    )

    logging.info(
        "- alifestd_mark_num_mrca_pairs_polars: counting mrca pairs...",
    )
    num_mrca_pairs = _alifestd_mark_num_mrca_pairs_asexual_fast_path(
        ancestor_ids,
    )

    return phylogeny_df.with_columns(
        pl.Series(name=mark_as, values=num_mrca_pairs),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Add column with number of leaf pairs whose most recent common ancestor is each node.

Data is assumed to be in alife standard format.

Additional Notes
================
- Requires asexual phylogeny with contiguous ids and topologically sorted rows.

- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.

See Also
========
phyloframe.legacy._alifestd_mark_num_mrca_pairs_asexual :
    CLI entrypoint for Pandas-based implementation.
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
        dfcli_module="phyloframe.legacy._alifestd_mark_num_mrca_pairs_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="num_mrca_pairs",
        type=str,
        help="output column name (default: num_mrca_pairs)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_num_mrca_pairs_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_num_mrca_pairs_polars,
                    mark_as=args.mark_as,
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
