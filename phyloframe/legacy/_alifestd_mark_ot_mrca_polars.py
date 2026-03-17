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
from ._alifestd_has_multiple_roots_polars import (
    alifestd_has_multiple_roots_polars,
)
from ._alifestd_is_topologically_sorted_polars import (
    alifestd_is_topologically_sorted_polars,
)
from ._alifestd_mark_leaves_polars import alifestd_mark_leaves_polars
from ._alifestd_mark_ot_mrca import _calc_ot_mrca_contiguous
from ._alifestd_try_add_ancestor_id_col_polars import (
    alifestd_try_add_ancestor_id_col_polars,
)


def alifestd_mark_ot_mrca_polars(
    phylogeny_df: pl.DataFrame,
    mark_as: str = "ot_mrca",
) -> pl.DataFrame:
    """Appends columns characterizing the Most Recent Common Ancestor
    (MRCA) of the entire extant population at each taxon's
    `origin_time`.

    The output column name prefix can be changed via the ``mark_as`` parameter.

    The extant population is defined in terms of active lineages: any
    branch of the tree existing at an `origin_time` which contains at
    least one descendant at or after that time.

    New Columns
    -----------
    ot_mrca_id : int
        The unique identifier of the MRCA for the population that was
        extant at this organism's `origin_time`.

    ot_mrca_time_of : int or float
        The `origin_time` of that MRCA.

    ot_mrca_time_since : int or float
        The duration elapsed between the MRCA's `origin_time` and this
        taxon's `origin_time`.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        The phylogeny as a dataframe in alife standard format.

        Must represent an asexual phylogeny with a single root.

    Returns
    -------
    polars.DataFrame
        The phylogeny with added `ot_mrca_id`, `ot_mrca_time_of`, and
        `ot_mrca_time_since` columns.

    See Also
    --------
    alifestd_mark_ot_mrca_asexual :
        Pandas-based implementation.
    """
    phylogeny_df = phylogeny_df.lazy().collect()

    # handle empty case
    if phylogeny_df.is_empty():
        return phylogeny_df.with_columns(
            pl.lit(0).cast(pl.Int64).alias(f"{mark_as}_id"),
            pl.lit(0).cast(pl.Float64).alias(f"{mark_as}_time_of"),
            pl.lit(0).cast(pl.Float64).alias(f"{mark_as}_time_since"),
        )

    # setup
    phylogeny_df = alifestd_try_add_ancestor_id_col_polars(phylogeny_df)

    if alifestd_has_multiple_roots_polars(phylogeny_df):
        raise NotImplementedError("multiple roots not supported")

    if not alifestd_is_topologically_sorted_polars(phylogeny_df):
        raise NotImplementedError("topological sort not yet supported")

    phylogeny_df = alifestd_mark_leaves_polars(phylogeny_df)

    if not alifestd_has_contiguous_ids_polars(phylogeny_df):
        raise NotImplementedError("non-contiguous ids not yet supported")

    # extract numpy arrays for fast sequential processing
    ids = phylogeny_df["id"].to_numpy()
    ancestor_ids = phylogeny_df["ancestor_id"].to_numpy()
    origin_times = phylogeny_df["origin_time"].to_numpy()
    is_leaf = phylogeny_df["is_leaf"].to_numpy()

    ot_mrca_id, ot_mrca_time_of, ot_mrca_time_since = _calc_ot_mrca_contiguous(
        ids, ancestor_ids, origin_times, is_leaf
    )

    return phylogeny_df.with_columns(
        pl.Series(f"{mark_as}_id", ot_mrca_id),
        pl.Series(f"{mark_as}_time_of", ot_mrca_time_of),
        pl.Series(f"{mark_as}_time_since", ot_mrca_time_since),
    )


_raw_description = f"""{os.path.basename(__file__)} | (phyloframe v{get_phyloframe_version()}/joinem v{joinem.__version__})

Append columns characterizing the Most Recent Common Ancestor (MRCA) of the entire extant population at each taxon's `origin_time`.

Data is assumed to be in alife standard format.

Additional Notes
================
- Use `--eager-read` if modifying data file inplace.

- This CLI entrypoint is experimental and may be subject to change.
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
        dfcli_module="phyloframe.legacy._alifestd_mark_ot_mrca_polars",
        dfcli_version=get_phyloframe_version(),
    )
    parser.add_argument(
        "--mark-as",
        default="ot_mrca",
        type=str,
        help="output column name prefix (default: ot_mrca)",
    )
    return parser


if __name__ == "__main__":
    begin_prod_logging()

    parser = _create_parser()
    args, __ = parser.parse_known_args()

    try:
        with log_context_duration(
            "phyloframe.legacy._alifestd_mark_ot_mrca_polars",
            logging.info,
        ):
            _run_dataframe_cli(
                base_parser=parser,
                output_dataframe_op=functools.partial(
                    alifestd_mark_ot_mrca_polars, mark_as=args.mark_as
                ),
            )
    except NotImplementedError as e:
        logging.error(
            "- polars op not yet implemented, use pandas op CLI instead",
        )
        raise e
